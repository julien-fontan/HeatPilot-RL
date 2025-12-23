import gymnasium as gym
from gymnasium import spaces
import numpy as np

# On importe la Dataclass et le moteur optimisé
from district_heating_model import PipeConfig, DistrictHeatingNetwork
from utils import generate_smooth_profile
from graph_utils import Graph
import config

class HeatNetworkEnv(gym.Env):
    """
    Environnement Gym optimisé pour le contrôle du réseau de chaleur.
    Compatible avec le moteur vectorisé DistrictHeatingNetwork.
    """
    def __init__(self):
        super(HeatNetworkEnv, self).__init__()

        # --- Paramètres de Simulation ---
        self.dt = config.TRAINING_PARAMS["dt"]
        self.t_max = config.SIMULATION_PARAMS["t_max_day"]
        self.dx = config.SIMULATION_PARAMS["dx"]
        self.max_steps = config.TRAINING_PARAMS["episode_length_steps"]

        self.props = config.PHYSICAL_PROPS

        # --- Contraintes Physiques ---
        self.temp_min = config.CONTROL_PARAMS["temp_min"]
        self.temp_max = config.CONTROL_PARAMS["temp_max"]
        self.flow_min = config.CONTROL_PARAMS["flow_min"]
        self.flow_max = config.CONTROL_PARAMS["flow_max"]
        
        self.max_temp_rise = config.CONTROL_PARAMS["max_temp_rise_per_dt"]
        self.max_temp_drop = config.CONTROL_PARAMS["max_temp_drop_per_dt"]
        self.max_flow_var = config.CONTROL_PARAMS["max_flow_delta_per_dt"]
        self.enable_ramps = config.CONTROL_PARAMS.get("enable_ramps", True)

        # --- Analyse de la Topologie ---
        self.edges = config.EDGES
        self.graph = Graph(self.edges)
        
        all_children = self.graph.get_child_nodes()
        self.branching_nodes = self.graph.get_branching_nodes()
        self.branching_map = {n: all_children[n] for n in self.branching_nodes}
        
        self.inlet_node = self.graph.get_inlet_node()
        self.consumer_nodes = self.graph.get_consumer_nodes()

        # Dimensions pour l'observation
        self.n_nodes = self.graph.get_nodes_count()
        self.n_pipes = len(self.edges)
        self.n_consumers = len(self.consumer_nodes)

        # --- Espaces d'Action ---
        # [T_target, Flow_target, Split_1, Split_2...]
        n_actions = 2 + len(self.branching_nodes)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_actions,), dtype=np.float32)

        # --- Espaces d'Observation (Vision Totale) ---
        # [T_tous_les_noeuds] + [Flow_tous_les_tuyaux] + [Demandes_consommateurs]
        
        high_obs = np.concatenate([
            [config.CONTROL_PARAMS["temp_max"]] * self.n_nodes,
            [config.CONTROL_PARAMS["flow_max"]] * self.n_pipes,
            [config.POWER_PROFILE_CONFIG["p_max"]] * self.n_consumers
        ]).astype(np.float32)
        
        low_obs = np.full_like(high_obs, 0)
        
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        self.network = None
        self.state = None
        self.current_t = 0.0
        self.demand_funcs = {}
        
        self.actual_inlet_temp = config.SIMULATION_PARAMS["initial_temp"]
        self.actual_mass_flow = config.SIMULATION_PARAMS["initial_flow"]

        # KPIs pour render/reward
        self.last_total_p_demand = 0.0
        self.last_total_p_supplied = 0.0
        self.last_p_source = 0.0
        self.last_p_pump = 0.0
        self.step_count = 0

    def reset(self, seed=None, options=None):
        if seed is None:
            seed = config.GLOBAL_SEED
        super().reset(seed=seed)

        # 1. Génération des configs de tuyaux (PipeConfig)
        lengths, diameters, n_segments, h_vals = PipeConfig.generate_parameters(
            edges=self.edges, dx=self.dx, seed=seed
        )

        pipe_configs = []
        for i, (u, v) in enumerate(self.edges):
            pc = PipeConfig(
                nodes=(u, v),
                length=float(lengths[i]),
                diameter=float(diameters[i]),
                dx=self.dx,
                rho=self.props["rho"],
                cp=self.props["cp"],
                heat_loss_coeff=float(h_vals[i]),
                thermal_conductivity=self.props["thermal_conductivity"],
                external_temp=self.props["external_temp"]
            )
            pipe_configs.append(pc)

        # 2. Demandes
        rng = np.random.default_rng(seed)
        self.demand_funcs = {}
        for node in self.consumer_nodes:
            self.demand_funcs[node] = generate_smooth_profile(
                t_end=self.t_max,
                step_time=config.POWER_PROFILE_CONFIG["step_time"],
                min_val=config.POWER_PROFILE_CONFIG["p_min"],
                max_val=config.POWER_PROFILE_CONFIG["p_max"],
                seed=rng.integers(0, 1_000_000)
            )

        # 3. Construction du Réseau
        # # On initialise avec une température aléatoire chaude pour aider l'apprentissage (Warm Start)
        # init_temp = rng.uniform(60.0, 80.0) 
        
        self.network = DistrictHeatingNetwork(
            graph=self.graph,
            pipe_configs=pipe_configs,
            t_min_return=config.SIMULATION_PARAMS["min_return_temp"],
            inlet_temp=config.SIMULATION_PARAMS["initial_temp"],
            inlet_mass_flow=config.SIMULATION_PARAMS["initial_flow"],
            rho=self.props["rho"],
            cp=self.props["cp"],
            node_power_funcs=self.demand_funcs
        )

        # 4. État initial (Toutes les cellules à T_init)
        self.state = np.full(self.network.total_cells, config.SIMULATION_PARAMS["initial_temp"])
        self.current_t = 0.0
        self.actual_inlet_temp = config.SIMULATION_PARAMS["initial_temp"]
        self.actual_mass_flow = config.SIMULATION_PARAMS["initial_flow"]
        self.step_count = 0

        # Calcul obs initiales (bizarre)
        pipe_flows = self.network._compute_mass_flows(0.0)
        node_temps = self.network._solve_nodes_temperature(self.state, pipe_flows, 0.0)

        return self._get_obs(node_temps, pipe_flows), {}
    
    def step(self, action):
        # --- 1. Décodage de l'action ---
        # Température
        raw_temp = np.clip(action[0], -1.0, 1.0)
        target_temp = self.temp_min + 0.5 * (raw_temp + 1.0) * (self.temp_max - self.temp_min)

        # Débit
        raw_flow = np.clip(action[1], -1.0, 1.0)
        target_flow = self.flow_min + 0.5 * (raw_flow + 1.0) * (self.flow_max - self.flow_min)

        # Splits
        split_actions = 0.5 * (action[2:] + 1.0) # -> [0, 1]

        # --- 2. Application des rampes ---
        if self.enable_ramps:
            # Rampe Température
            delta_T = target_temp - self.actual_inlet_temp
            delta_T = np.clip(delta_T, -self.max_temp_drop, self.max_temp_rise)
            self.actual_inlet_temp += delta_T
            
            # Rampe Débit
            delta_F = target_flow - self.actual_mass_flow
            delta_F = np.clip(delta_F, -self.max_flow_var, self.max_flow_var)
            self.actual_mass_flow += delta_F
        else:
            self.actual_inlet_temp = target_temp
            self.actual_mass_flow = target_flow

        # Clipping final sécurité
        self.actual_inlet_temp = np.clip(self.actual_inlet_temp, self.temp_min, self.temp_max)
        self.actual_mass_flow = np.clip(self.actual_mass_flow, self.flow_min, self.flow_max)

        # --- 3. Configuration du réseau ---
        # Mise à jour des splits
        node_splits = {}
        for i, node in enumerate(self.branching_nodes):
            frac = np.clip(split_actions[i], 0.01, 0.99)
            children = self.branching_map[node]
            if len(children) >= 2:
                node_splits[node] = {children[0]: frac, children[1]: 1.0 - frac}
        
        self.network.set_node_splits(node_splits)
        self.network.inlet_mass_flow = self.actual_mass_flow
        self.network.inlet_temp = self.actual_inlet_temp

        # --- 4. Simulation physique (solve_ivp) ---
        t_next = self.current_t + self.dt
        sol = self.network.solve(
            (self.current_t, t_next),
            self.state,
            method="RK45",
            rtol=config.SIMULATION_PARAMS["rtol"],
            atol=config.SIMULATION_PARAMS["atol"]
        )
        
        self.state = sol.y[:, -1]
        self.current_t = t_next
        self.step_count += 1

        # --- 5. Calcul des métriques pour reward & observation ---
        # A. Débits dans les tuyaux
        pipe_flows = self.network._compute_mass_flows(self.current_t)
        
        # B. Températures aux noeuds
        node_temps = self.network._solve_nodes_temperature(self.state, pipe_flows, self.current_t)
        
        # C. Calcul puissance fournie vs demandée
        parents_map = self.graph.get_parent_nodes()
        total_demand = 0.0
        total_supplied = 0.0
        supplied_per_node = {}
        demand_per_node = {}
        
        for node_id in self.consumer_nodes:
            p_target = self.demand_funcs[node_id](self.current_t)
            demand_per_node[node_id] = p_target
            total_demand += p_target
            
            # Récup débit entrant
            parents = parents_map.get(node_id, [])
            m_in = 0.0
            for p_node in parents:
                edge = self.graph.edges[(p_node, node_id)]
                m_in += pipe_flows[edge["pipe_index"]]
            
            # Calcul puissance physique
            if m_in > 1e-9:
                node_idx = self.graph.get_id_from_node(node_id)
                T_in = node_temps[node_idx]
                delta_T_max = max(T_in - config.SIMULATION_PARAMS["min_return_temp"], 0.0)
                p_max = m_in * self.props["cp"] * delta_T_max
                p_eff = min(p_target, p_max)
            else:
                p_eff = 0.0
            
            total_supplied += p_eff
            supplied_per_node[node_id] = p_eff

        # Coûts source/pompe
        p_source = self.actual_mass_flow * self.props["cp"] * (self.actual_inlet_temp - config.SIMULATION_PARAMS["min_return_temp"])
        p_pump = 1000.0 * self.actual_mass_flow # Simplifié

        self.last_total_p_demand = total_demand
        self.last_total_p_supplied = total_supplied
        self.last_p_source = p_source
        self.last_p_pump = p_pump

        # --- 6. Calcul reward ---
        reward = self._compute_reward(total_demand, total_supplied, p_source, p_pump)

        # --- 7. Info dictionary (pour evaluate) ---
        info = {
            "pipe_mass_flows": pipe_flows,            
            "node_temperatures": node_temps,
            "demand_per_node": demand_per_node,
            "supplied_per_node": supplied_per_node,
        }
        
        # --- 8. Sortie ---
        obs = self._get_obs(node_temps, pipe_flows) 
        terminated = False
        truncated = (self.current_t >= self.t_max) or (self.step_count >= self.max_steps)

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, demand, supplied, source_power, pump_power):
        """Logique de récompense séparée."""
        rc = config.REWARD_PARAMS
        w = rc["weights"]
        p = rc["params"]

        # Conversion kW
        dem_kw = demand / 1000.0
        sup_kw = supplied / 1000.0
        src_kw = source_power / 1000.0
        pump_kw = pump_power / 1000.0

        # 1. Confort (pénalité sous-production)
        under_supply = max(0.0, dem_kw - sup_kw)
        r_conf = -w["comfort"] * (under_supply / p["p_ref"])

        # 2. Sobriété prod (pénalité surproduction inutile)
        # On pénalise si on produit plus que la demande (pertes réseau incluses implicitement si T_return > T_min)
        excess = max(0.0, src_kw - dem_kw)
        r_prod = -w["boiler"] * (excess / p["p_ref"])

        # 3. Pompe
        diff_pump = (pump_kw - p["p_pump_nominal"]) / p["p_pump_nominal"]
        r_pump = -w["pump"] * (diff_pump**2 + max(0.0, diff_pump))

        return float(r_conf + r_prod + r_pump)  # check(env) exige un reward au format float

    def _get_obs(self, node_temps, pipe_flows):
        """
        Construit le vecteur d'observation complet.
        Args:
            node_temps: np.array de toutes les températures nodales
            pipe_flows: np.array de tous les débits tuyaux
        """
        demands = [self.demand_funcs[n](self.current_t) for n in self.consumer_nodes]

        # Concaténation [T_nodes, Flows, Demands]
        obs = np.concatenate([
            node_temps,
            pipe_flows,
            demands
        ], dtype=np.float32)
        
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def render(self):
        print(
            f"T={self.current_t:.1f}s | "
            f"In: {self.actual_inlet_temp:.1f}°C, {self.actual_mass_flow:.1f}kg/s | "
            f"Dem: {self.last_total_p_demand/1000:.0f}kW | "
            f"Sup: {self.last_total_p_supplied/1000:.0f}kW | "
            f"Src: {self.last_p_source/1000:.0f}kW"
        )