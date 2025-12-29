import gymnasium as gym
from gymnasium import spaces
import numpy as np

from district_heating_model import PipeConfig, DistrictHeatingNetwork, create_pipe_configs
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

        # --- Warmup ---
        self.warmup_enabled = config.TRAINING_PARAMS.get("warmup_enabled", False)
        self.warmup_duration = config.SIMULATION_PARAMS.get("warmup_duration", 0)
        self.warmup_temp = config.SIMULATION_PARAMS.get("initial_temp")
        self.warmup_flow = config.SIMULATION_PARAMS.get("initial_flow")

        self.props = config.PHYSICAL_PROPS

        # --- Contraintes Physiques & Rampes ---
        self.temp_min = config.CONTROL_PARAMS["temp_min"]
        self.temp_max = config.CONTROL_PARAMS["temp_max"]
        self.flow_min = config.CONTROL_PARAMS["flow_min"]
        self.flow_max = config.CONTROL_PARAMS["flow_max"]
        
        self.max_temp_rise = config.CONTROL_PARAMS["max_temp_rise_per_dt"]
        self.max_temp_drop = config.CONTROL_PARAMS["max_temp_drop_per_dt"]
        self.max_flow_var = config.CONTROL_PARAMS["max_flow_delta_per_dt"]
        self.max_split_speed = config.CONTROL_PARAMS.get("max_split_change_per_dt", 0.05)

        # --- Topologie ---
        self.edges = config.EDGES
        self.graph = Graph(self.edges)
        
        all_children = self.graph.get_child_nodes()
        self.branching_nodes = self.graph.get_branching_nodes()
        self.branching_map = {n: all_children[n] for n in self.branching_nodes}
        
        self.inlet_node = self.graph.get_inlet_node()
        self.consumer_nodes = self.graph.get_consumer_nodes()

        self.n_nodes = self.graph.get_nodes_count()
        self.n_pipes = len(self.edges)
        self.n_consumers = len(self.consumer_nodes)

        # Initialisation à 0.5 (50/50) au départ
        self.current_split_ratios = np.full(len(self.branching_nodes), 0.5, dtype=np.float32)


        # --- Espaces d'Action (Delta T, Delta Flow, Splits...) ---
        n_actions = 2 + len(self.branching_nodes)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_actions,), dtype=np.float32)

        # --- Espaces d'Observation ---
        high_obs = np.concatenate([
            [config.CONTROL_PARAMS["temp_max"]] * self.n_nodes,
            [config.CONTROL_PARAMS["flow_max"]] * self.n_pipes,
            [config.POWER_PROFILE_CONFIG["p_max"]] * self.n_consumers
        ]).astype(np.float32)
        
        low_obs = np.full_like(high_obs, 0)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # State vars
        self.network = None
        self.state = None
        self.current_t = 0.0
        self.demand_funcs = {}
        
        self.actual_inlet_temp = config.SIMULATION_PARAMS["initial_temp"]
        self.actual_mass_flow = config.SIMULATION_PARAMS["initial_flow"]
        self.step_count = 0
        
        # Optionnel: Désactiver la randomisation géométrique pour accélérer le reset
        self.randomize_geometry = config.SIMULATION_PARAMS.get("randomize_geometry", False)

    def reset(self, seed=None, options=None, node_splits=None):
        if seed is None: seed = config.GLOBAL_SEED
        super().reset(seed=seed)
        
        rng = np.random.default_rng(seed)

        # 1. Génération Demandes
        self.demand_funcs = {}
        for node in self.consumer_nodes:
            self.demand_funcs[node] = generate_smooth_profile(
                t_end=self.t_max,
                step_time=config.POWER_PROFILE_CONFIG["step_time"],
                min_val=config.POWER_PROFILE_CONFIG["p_min"],
                max_val=config.POWER_PROFILE_CONFIG["p_max"],
                seed=rng.integers(0, 1_000_000)
            )

        # --- CORRECTION 1 : Chargement des Splits par défaut SYSTEMATIQUE ---
        # On le fait ici pour que ce soit dispo pour la création OU la réutilisation
        if node_splits is None:
            node_splits = getattr(config, "DEFAULT_NODE_SPLITS", {})

        # 2. Gestion du Réseau
        if self.network is None or self.randomize_geometry:
            lengths, diameters, n_segments, h_vals = PipeConfig.generate_parameters(
                self.edges, self.dx, seed
            )
            pipe_configs = create_pipe_configs(
                self.edges, lengths, diameters, h_vals, self.dx, self.props
            )
            
            self.network = DistrictHeatingNetwork(
                graph=self.graph,
                pipe_configs=pipe_configs,
                t_min_return=config.SIMULATION_PARAMS["min_return_temp"],
                inlet_temp=config.SIMULATION_PARAMS["initial_temp"],
                inlet_mass_flow=config.SIMULATION_PARAMS["initial_flow"],
                rho=self.props["rho"],
                cp=self.props["cp"],
                node_power_funcs=self.demand_funcs,
                node_splits=node_splits # Initialisation physique correcte
            )
            self.state = np.full(self.network.total_cells, config.SIMULATION_PARAMS["initial_temp"])
        else:
            # Réutilisation
            self.network.update_power_profiles(self.demand_funcs)
            self.state = self.network.reset_state(config.SIMULATION_PARAMS["initial_temp"])
            # --- CORRECTION 2 : Forcer le reset des vannes physiques ---
            self.network.set_node_splits(node_splits)

        # 3. Reset Variables de contrôle
        self.current_t = 0.0
        self.actual_inlet_temp = config.SIMULATION_PARAMS["initial_temp"]
        self.actual_mass_flow = config.SIMULATION_PARAMS["initial_flow"]
        
        # --- CORRECTION 3 : Synchronisation de la mémoire de l'agent ---
        # On lit la config pour initialiser le vecteur de l'agent
        # Attention à l'ordre des enfants ! Il doit matcher celui de step()
        self.current_split_ratios = np.zeros(len(self.branching_nodes), dtype=np.float32)
        
        for i, node in enumerate(self.branching_nodes):
            children = self.branching_map[node] # [child1, child2]
            if len(children) >= 2:
                child1 = children[0]
                # On récupère la valeur dans la config (défaut 0.5 si introuvable)
                if node in node_splits and child1 in node_splits[node]:
                    val = node_splits[node][child1]
                else:
                    val = 0.5
                self.current_split_ratios[i] = val
            else:
                self.current_split_ratios[i] = 0.5 # Fallback (ne devrait pas arriver sur un branching node)

        self.step_count = 0
        
        self.network.inlet_temp = self.actual_inlet_temp
        self.network.inlet_mass_flow = self.actual_mass_flow

        # 4. Warmup
        if self.warmup_enabled and self.warmup_duration > 0:
            self._run_warmup()

        # 5. Obs Initiale
        pipe_flows = self.network._compute_mass_flows(self.current_t)
        node_temps = self.network._solve_nodes_temperature(self.state, pipe_flows, self.current_t)

        return self._get_obs(node_temps, pipe_flows), {}

    def _run_warmup(self):
        """Préchauffe le réseau pour éviter les transitoires violents au début."""
        self.network.inlet_temp = self.warmup_temp
        self.network.inlet_mass_flow = self.warmup_flow
        
        # On fait tourner la simu en boucle ouverte
        for _ in range(int(self.warmup_duration/self.dt)):
            t_next = self.current_t + self.dt
            sol = self.network.solve((self.current_t, t_next), self.state, method="RK45")
            self.state = sol.y[:, -1]
            self.current_t = t_next
            
        # Rétablissement des conditions contrôlables
        self.network.inlet_temp = self.actual_inlet_temp
        self.network.inlet_mass_flow = self.actual_mass_flow

    def step(self, action):
        # --- 1. Décodage Action ---
        # Action 0: Delta Température (Relatif à la rampe max)
        delta_temp_pct = np.clip(action[0], -1.0, 1.0)
        delta_T = delta_temp_pct * (self.max_temp_rise if delta_temp_pct >= 0 else self.max_temp_drop)
        self.actual_inlet_temp = np.clip(self.actual_inlet_temp + delta_T, self.temp_min, self.temp_max)

        # Action 1: Delta Débit (Relatif)
        delta_flow_pct = np.clip(action[1], -1.0, 1.0)
        delta_F = delta_flow_pct * self.max_flow_var
        self.actual_mass_flow = np.clip(self.actual_mass_flow + delta_F, self.flow_min, self.flow_max)

        # Action 2+: Vannes (Absolu [0, 1])
        # action[2:] contrôle la VITESSE de la vanne, pas sa position
        raw_split_deltas = action[2:]
        # split += action * vitesse_max
        # Si action est 0, la vanne ne bouge pas (elle reste à sa position précédente)
        delta_splits = np.clip(raw_split_deltas, -1.0, 1.0) * self.max_split_speed
        self.current_split_ratios += delta_splits
        self.current_split_ratios = np.clip(self.current_split_ratios, 0.05, 0.95)

        # split_actions = 0.5 * (action[2:] + 1.0) # (ancien)

        # --- 2. Mise à jour Réseau ---
        node_splits = {}
        for i, node in enumerate(self.branching_nodes):
            frac = self.current_split_ratios[i]
            children = self.branching_map[node]
            if len(children) >= 2:
                node_splits[node] = {children[0]: frac, children[1]: 1.0 - frac}
        
        self.network.set_node_splits(node_splits)
        self.network.inlet_mass_flow = self.actual_mass_flow
        self.network.inlet_temp = self.actual_inlet_temp

        # --- 3. Simulation Physique ---
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

        # --- 4. Métriques ---
        pipe_flows = self.network._compute_mass_flows(self.current_t)
        node_temps = self.network._solve_nodes_temperature(self.state, pipe_flows, self.current_t)
        node_temps_after = self.network._apply_node_power_consumption(self.current_t, node_temps, pipe_flows)

        # Calcul Demande vs Fourni
        parents_map = self.graph.get_parent_nodes()
        total_demand = 0.0
        total_supplied = 0.0
        
        for node_id in self.consumer_nodes:
            p_target = self.demand_funcs[node_id](self.current_t)
            total_demand += p_target
            
            # Récupération débit local
            m_in = 0.0
            for p_node in parents_map.get(node_id, []):
                edge = self.graph.edges[(p_node, node_id)]
                m_in += pipe_flows[edge["pipe_index"]]
            
            if m_in > 1e-9:
                node_idx = self.graph.get_id_from_node(node_id)
                T_in = node_temps[node_idx]
                delta_T_avail = max(T_in - config.SIMULATION_PARAMS["min_return_temp"], 0.0)
                p_max = m_in * self.props["cp"] * delta_T_avail
                total_supplied += min(p_target, p_max)

        # KPIs Source/Pompe
        p_source = self.actual_mass_flow * self.props["cp"] * (self.actual_inlet_temp - config.SIMULATION_PARAMS["min_return_temp"])
        p_pump = 1000.0 * self.actual_mass_flow 

        self.last_total_p_demand = total_demand
        self.last_total_p_supplied = total_supplied
        self.last_p_source = p_source
        self.last_p_pump = p_pump

        # --- 5. Reward & Info ---
        reward = self._compute_reward(total_demand, total_supplied, p_pump, node_temps_after, action)
        
        info = {
            "pipe_mass_flows": pipe_flows,            
            "node_temperatures": node_temps
        }
        
        terminated = False
        truncated = (self.current_t >= self.t_max) or (self.step_count >= self.max_steps)

        return self._get_obs(node_temps, pipe_flows), reward, terminated, truncated, info

    def _compute_reward(self, demand, supplied, pump_power, node_temps_after, action_vector):
        rc = config.REWARD_PARAMS
        w = rc["weights"]
        p = rc["params"]

        dem_kw = demand / 1000.0
        sup_kw = supplied / 1000.0
        pump_kw = pump_power / 1000.0

        # 1. Confort
        under_supply = max(0.0, dem_kw - sup_kw)
        r_conf = -w["comfort"] * (under_supply / p["p_ref"])

        # 2. Sobriété (Température de retour excessive aux terminaux)
        min_ret = config.SIMULATION_PARAMS["min_return_temp"]
        terminal_nodes = self.graph.get_terminal_nodes()
        excess_temp = 0.0
        if terminal_nodes:
            sum_excess = 0.0
            for nid in terminal_nodes:
                idx = self.graph.get_id_from_node(nid)
                t_val = node_temps_after[idx]
                if t_val > min_ret:
                    sum_excess += (t_val - min_ret)
            excess_temp = sum_excess / len(terminal_nodes)
            
        r_prod = -w["boiler"] * excess_temp / min_ret

        # 3. Pompe
        diff_pump = (pump_kw - p["p_pump_nominal"]) / p["p_pump_nominal"]
        r_pump = -w["pump"] * (diff_pump**2 + max(0.0, diff_pump))

        # 4. STABILITÉ (NOUVEAU)
        # On veut que l'agent fasse des petits changements (Deltas proches de 0).
        # action[0] est le Delta Température (entre -1 et 1)
        # action[1] est le Delta Débit (entre -1 et 1)
        # action[2:] sont les Positions Vannes (Splits). Pour eux, c'est plus dur de punir        
        # Pénalité quadratique sur la "violence" de la commande source
        instability = np.sum(np.square(action_vector))
        
        # Pour les splits, on peut pénaliser les valeurs extrêmes (0 ou 1) si on veut du mélange
        # ou simplement laisser faire. Commençons par stabiliser la source.
        
        r_stab = -w.get("stability", 0.0) * instability

        return float(r_conf + r_prod + r_pump)

    def _get_obs(self, node_temps, pipe_flows):
        demands = [self.demand_funcs[n](self.current_t) for n in self.consumer_nodes]
        obs = np.concatenate([node_temps, pipe_flows, demands], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def render(self):
        print(f"T={self.current_t:.0f} | In: {self.actual_inlet_temp:.1f}C, {self.actual_mass_flow:.1f}kg/s | "
              f"Dem: {self.last_total_p_demand/1e3:.0f}kW | Sup: {self.last_total_p_supplied/1e3:.0f}kW")