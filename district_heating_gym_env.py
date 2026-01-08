import gymnasium as gym
from gymnasium import spaces
import numpy as np

from district_heating_model import DistrictHeatingNetwork, PipeConfig, create_pipe_configs
from utils import generate_smooth_profile
from graph_utils import Graph
import config

class HeatNetworkEnv(gym.Env):
    def __init__(self, fixed_seed=None):
        super().__init__()
        self.fixed_seed = fixed_seed
        self.dt = config.TRAINING_PARAMS["dt"]
        self.t_max = config.SIMULATION_PARAMS["t_max_day"]
        self.dx = config.SIMULATION_PARAMS["dx"]
        self.max_steps = config.TRAINING_PARAMS["episode_length_steps"]
        self.props = config.PHYSICAL_PROPS

        self.edges = config.EDGES
        self.graph = Graph(self.edges)
        
        self.branching_nodes = self.graph.branching_nodes
        self.agent_manages_splits = True # Par défaut l'agent gère les vannes

        # Espaces d'Action
        n_actions = 2 + len(self.branching_nodes)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_actions,), dtype=np.float32)

        # Espaces d'Observation
        n_obs = self.graph.n_nodes + len(self.edges) + len(self.graph.consumer_nodes)
        high_obs = np.concatenate([
            [config.CONTROL_PARAMS["temp_max"]] * self.graph.n_nodes,
            [config.CONTROL_PARAMS["flow_max"]] * len(self.edges),
            [config.POWER_PROFILE_CONFIG["p_max"]] * len(self.graph.consumer_nodes)
        ]).astype(np.float32)
        
        low_obs = np.full_like(high_obs, 0)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, shape=(n_obs,), dtype=np.float32)

        self.network = None
        self.randomize_geometry = config.SIMULATION_PARAMS.get("randomize_geometry", False)
        self.current_split_ratios = np.full(len(self.branching_nodes), 0.5, dtype=np.float32)

    def reset(self, seed=None, options=None, node_splits=None):
        if seed is None and self.fixed_seed is not None:
             seed = self.fixed_seed
        
        super().reset(seed=seed)
        # Utilisation du générateur de nombres aléatoires de Gym (correctement initialisé par super().reset())
        rng = self.np_random

        self.demand_funcs = {}
        for node in self.graph.consumer_nodes:
            self.demand_funcs[node] = generate_smooth_profile(
                t_end=self.t_max,
                step_time=config.POWER_PROFILE_CONFIG["step_time"],
                min_val=config.POWER_PROFILE_CONFIG["p_min"],
                max_val=config.POWER_PROFILE_CONFIG["p_max"],
                seed=rng.integers(0, 1_000_000)
            )

        if node_splits is None:
            node_splits = getattr(config, "DEFAULT_NODE_SPLITS", {})

        if self.network is None or self.randomize_geometry:
            # Note: PipeConfig.generate_parameters attend une seed int.
            # Geo seed est fixe pour avoir toujours le même réseau physique
            geo_seed = config.GLOBAL_SEED
            lengths, diameters, n_segments, h_vals = PipeConfig.generate_parameters(self.edges, self.dx, geo_seed)
            pipe_configs = create_pipe_configs(self.edges, lengths, diameters, h_vals, self.dx, self.props)
            self.network = DistrictHeatingNetwork(
                graph=self.graph,
                pipe_configs=pipe_configs,
                t_min_return=config.SIMULATION_PARAMS["min_return_temp"],
                inlet_temp=config.SIMULATION_PARAMS["initial_temp"],
                inlet_mass_flow=config.SIMULATION_PARAMS["initial_flow"],
                rho=self.props["rho"], cp=self.props["cp"],
                node_power_funcs=self.demand_funcs,
                node_splits=node_splits 
            )
            self.state = np.full(self.network.total_cells, config.SIMULATION_PARAMS["initial_temp"])
        else:
            self.network.update_power_profiles(self.demand_funcs)
            self.state = self.network.reset_state(config.SIMULATION_PARAMS["initial_temp"])
            self.network.set_node_splits(node_splits)

        self.current_t = 0.0
        self.actual_inlet_temp = config.SIMULATION_PARAMS["initial_temp"]
        self.actual_mass_flow = config.SIMULATION_PARAMS["initial_flow"]
        self.step_count = 0
        self.current_split_ratios.fill(0.5)

        if config.TRAINING_PARAMS.get("warmup_enabled", False):
            self._run_warmup()

        metrics = self.network.get_instant_metrics(self.current_t, self.state)
        return self._make_obs(metrics), {}

    def _run_warmup(self):
        duration = config.SIMULATION_PARAMS.get("warmup_duration", 0)
        steps = int(duration / self.dt)
        for _ in range(steps):
            t_next = self.current_t + self.dt
            sol = self.network.solve((self.current_t, t_next), self.state, method="RK45")
            self.state = sol.y[:, -1]
            self.current_t = t_next
        self.network.inlet_temp = self.actual_inlet_temp
        self.network.inlet_mass_flow = self.actual_mass_flow

    def step(self, action):
        dt_temp_pct, dt_flow_pct = action[0], action[1]
        
        self.actual_inlet_temp = np.clip(
            self.actual_inlet_temp + dt_temp_pct * config.CONTROL_PARAMS["max_temp_rise_per_dt"],
            config.CONTROL_PARAMS["temp_min"], config.CONTROL_PARAMS["temp_max"]
        )
        self.actual_mass_flow = np.clip(
            self.actual_mass_flow + dt_flow_pct * config.CONTROL_PARAMS["max_flow_delta_per_dt"],
            config.CONTROL_PARAMS["flow_min"], config.CONTROL_PARAMS["flow_max"]
        )

        if self.agent_manages_splits:
            delta_splits = action[2:] * config.CONTROL_PARAMS.get("max_split_change_per_dt", 0.05)
            self.current_split_ratios = np.clip(self.current_split_ratios + delta_splits, 0.01, 0.99)
            
            node_splits = {}
            for i, node in enumerate(self.branching_nodes):
                children = self.graph.child_nodes[node]
                if len(children) >= 2:
                    node_splits[node] = {children[0]: self.current_split_ratios[i], children[1]: 1.0 - self.current_split_ratios[i]}
            self.network.set_node_splits(node_splits)
        else:
            action[2:] = 0.0 
            splits_dict = self.network.compute_heuristic_splits(self.current_t)
            self.network.set_node_splits(splits_dict)
            
            for i, node in enumerate(self.branching_nodes):
                if node in splits_dict:
                     child0 = self.graph.child_nodes[node][0]
                     self.current_split_ratios[i] = splits_dict[node].get(child0, 0.5)

        self.network.inlet_temp = self.actual_inlet_temp
        self.network.inlet_mass_flow = self.actual_mass_flow

        t_next = self.current_t + self.dt
        sol = self.network.solve(
            (self.current_t, t_next), self.state, method="RK45",
            rtol=config.SIMULATION_PARAMS["rtol"], atol=config.SIMULATION_PARAMS["atol"]
        )
        self.state = sol.y[:, -1]
        self.current_t = t_next
        self.step_count += 1
        
        metrics = self.network.get_instant_metrics(self.current_t, self.state)
        reward = self._compute_reward(metrics, action)
        truncated = (self.current_t >= self.t_max)
        
        info = {
            "pipe_mass_flows": metrics["mass_flows"],            
            "node_temperatures": metrics["node_temperatures"]
        }
        return self._make_obs(metrics), reward, False, truncated, info

    def _compute_ideal_splits(self, t):
        splits_dict = self.network.compute_heuristic_splits(t)
        ideal_splits = np.full(len(self.branching_nodes), 0.5, dtype=np.float32)
        for i, node in enumerate(self.branching_nodes):
            if node in splits_dict:
                children = self.graph.child_nodes[node]
                if children:
                    child0 = children[0]
                    ideal_splits[i] = splits_dict[node].get(child0, 0.5)
        return ideal_splits

    def _compute_reward(self, metrics, action):
        rc = config.REWARD_PARAMS
        w, p = rc["weights"], rc["params"]

        dem_kw, sup_kw = metrics["demand"] / 1000.0, metrics["supplied"] / 1000.0
        wasted_kw, pump_kw = metrics["wasted"] / 1000.0, metrics["pump_power"] / 1000.0

        diff_supply = dem_kw - sup_kw
        r_supply_bonus = p.get("supply_bonus_amp", 1.0) * np.exp(- (diff_supply**2) / (p.get("sigma_supply", 100.0)**2))
        r_supply_linear = -1.0 * (max(0.0, diff_supply) / p["p_ref"])   # Sert à guider l'agent quand il est loin du confort
        term_comfort = w["comfort"] * (r_supply_linear + r_supply_bonus)

        r_waste_bonus = p.get("waste_bonus_amp", 1.0) * np.exp(- (wasted_kw**2) / (p.get("sigma_waste", 100.0)**2))
        r_waste_linear = -1.0 * (wasted_kw / p.get("p_ref", p["p_ref"]))    # Sert à guider l'agent quand il y a beaucoup de pertes
        term_sobriety = w["waste"] * (r_waste_linear + r_waste_bonus)

        is_comfort = (diff_supply <= p.get("combo_comfort_deficit_max_kw", 20.0))
        is_waste_ok = (wasted_kw <= p.get("combo_wasted_max_kw", 50.0))
        term_combo = p.get("combo_bonus", 0.0) if (is_comfort and is_waste_ok) else 0.0

        # Split heuristic bonus (Vectorized)
        ideal_splits = self._compute_ideal_splits(self.current_t)
        current_splits = self.current_split_ratios
        split_diffs = np.abs(current_splits - ideal_splits)
        
        # Enveloppe stricte : +2.0 si diff <= 5%, sinon pénalité douce
        r_splits_arr = np.where(split_diffs <= 0.05, 2.0, -0.5 * split_diffs)
        r_splits = np.mean(r_splits_arr) if r_splits_arr.size > 0 else 0.0
        term_splits = 4.0 * r_splits

        r_pump = w["pump"] * np.exp(-0.5 * ((pump_kw - p["p_pump_nominal"]) / p.get("p_pump_sigma", 0.8)) ** 2) # Bonus gaussien autour de la puissance nominale de la pompe

        return float(term_comfort + term_sobriety + term_combo + r_pump + term_splits)

    def _make_obs(self, metrics):
        demands = [self.demand_funcs[n](self.current_t) for n in self.graph.consumer_nodes]
        return np.concatenate([metrics["node_temperatures"], metrics["mass_flows"], demands], dtype=np.float32)

    def set_agent_split_control(self, enabled: bool):
        self.agent_manages_splits = enabled