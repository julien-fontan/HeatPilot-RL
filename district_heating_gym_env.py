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
        self.downstream_consumers_map = self.graph.get_downstream_consumers_map()
        
        # État du curriculum (piloté par le Callback ou la config)
        self.agent_manages_splits = not config.CURRICULUM_PARAMS.get("enable_heuristic_splits_at_start", False)

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
                node_splits=node_splits 
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
                self.current_split_ratios[i] = 0.5 

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

        if self.agent_manages_splits:
            # 1. Mode Agent : On utilise les deltas fournis par le réseau de neurones
            raw_split_deltas = action[2:]
            delta_splits = np.clip(raw_split_deltas, -1.0, 1.0) * self.max_split_speed
            self.current_split_ratios += delta_splits
            self.current_split_ratios = np.clip(self.current_split_ratios, 0.01, 0.99)
        else:
            # 2. Mode Heuristique : On force les valeurs calculées
            ideal = self._compute_ideal_splits(self.current_t)
            self.current_split_ratios = ideal
            # IMPORTANT : On écrase l'action de l'agent dans le vecteur 'action' 
            # pour ne pas le pénaliser sur la stabilité d'une action qu'il ne contrôle pas.
            action[2:] = 0.0

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

        # --- NOUVEAU : Calcul explicite du "Wasted" (Pertes aux terminaux) ---
        total_wasted = 0.0
        terminal_nodes = self.graph.get_terminal_nodes()
        min_ret = config.SIMULATION_PARAMS["min_return_temp"]
        
        for nid in terminal_nodes:
            idx = self.graph.get_id_from_node(nid)
            # Débit entrant dans le terminal
            m_in = 0.0
            for p_node in parents_map.get(nid, []):
                edge = self.graph.edges[(p_node, nid)]
                m_in += pipe_flows[edge["pipe_index"]]
            
            if m_in > 1e-9:
                # Température après consommation (sortie de sous-station)
                T_out = node_temps_after[idx]
                if T_out > min_ret:
                    # Puissance thermique rejetée inutilement
                    total_wasted += m_in * self.props["cp"] * (T_out - min_ret)

        # KPIs Source/Pompe
        p_source = self.actual_mass_flow * self.props["cp"] * (self.actual_inlet_temp - min_ret)
        p_pump = 1000.0 * self.actual_mass_flow 

        self.last_total_p_demand = total_demand
        self.last_total_p_supplied = total_supplied
        self.last_p_source = p_source
        self.last_p_pump = p_pump

        # --- 5. Reward & Info ---
        reward = self._compute_reward(total_demand, total_supplied, total_wasted, p_pump, action)
        
        info = {
            "pipe_mass_flows": pipe_flows,            
            "node_temperatures": node_temps
        }
        
        terminated = False
        truncated = (self.current_t >= self.t_max) or (self.step_count >= self.max_steps)

        return self._get_obs(node_temps, pipe_flows), reward, terminated, truncated, info

    def _compute_reward(self, demand, supplied, wasted, pump_power, action_vector):
        rc = config.REWARD_PARAMS
        w = rc["weights"]
        p = rc["params"]

        # Conversion en kW
        dem_kw = demand / 1000.0
        sup_kw = supplied / 1000.0
        wasted_kw = wasted / 1000.0
        pump_kw = pump_power / 1000.0

        # --- 1. GRATIFICATION PUISSANCE FOURNIE (Zone 100kW) ---
        diff_supply = dem_kw - sup_kw
        sigma_supply = 100.0 
        
        # Bonus exponentiel : récompense fortement la précision autour de 0
        r_supply_precision = 2.0 * np.exp(- (diff_supply**2) / (sigma_supply**2))
        
        # Pénalité linéaire de fond pour guider l'agent s'il est très loin
        r_supply_linear = -1.0 * (max(0.0, diff_supply) / p["p_ref"])
        
        term_comfort = w["comfort"] * (r_supply_linear + r_supply_precision)

        # --- 2. GRATIFICATION PUISSANCE PERDUE (Zone < 100kW) ---

        # - un bonus exponentiel (fort signal quand wasted ~ 0)
        # - une pénalité linéaire (garde un gradient même quand wasted est grand)
        sigma_waste = p.get("sigma_waste", 100.0)
        waste_ref = p.get("p_waste_ref", p["p_ref"])
        waste_bonus_amp = p.get("waste_bonus_amp", 1.0)

        r_waste_precision = waste_bonus_amp * np.exp(- (wasted_kw**2) / (sigma_waste**2))
        r_waste_linear = -1.0 * (wasted_kw / waste_ref)

        term_sobriety = w["boiler"] * (r_waste_linear + r_waste_precision)

        # --- 2bis. BONUS COMBO (Confort + Faibles pertes) ---
        # Bonus important si (a) le déficit de fourniture est faible (confort atteint)
        # ET (b) les pertes terminales sont sous un seuil.
        combo_bonus = p.get("combo_bonus", 0.0)
        comfort_deficit_max_kw = p.get("combo_comfort_deficit_max_kw", 20.0)
        wasted_max_kw = p.get("combo_wasted_max_kw", 50.0)

        # diff_supply = dem_kw - sup_kw (positif = déficit). Confort atteint si déficit <= seuil.
        is_comfort_ok = (diff_supply <= comfort_deficit_max_kw)
        is_wasted_ok = (wasted_kw <= wasted_max_kw)
        term_combo = combo_bonus if (is_comfort_ok and is_wasted_ok) else 0.0

        # --- 3. BONIFICATION SPLITS (Enveloppe 10% -> +/- 0.05) ---
        ideal_splits = self._compute_ideal_splits(self.current_t)
        current_splits = self.current_split_ratios
        
        # Calcul de l'écart absolu
        split_diffs = np.abs(current_splits - ideal_splits)
        
        r_splits = 0.0
        for diff in split_diffs:
            # Enveloppe stricte demandée
            if diff <= 0.05:
                r_splits += 2.0 # Gratification
            else:
                # Pénalité douce pour guider vers l'enveloppe
                r_splits -= 0.5 * diff 
        
        if len(split_diffs) > 0:
            r_splits /= len(split_diffs)
        
        term_splits = 4.0 * r_splits

        # --- 4. Pompe & Stabilité ---
        # Bonus explicite centré autour du débit nominal (≈ 15 kg/s).
        # Ici pump_kw == débit massique (car pump_power = 1000 * m_dot et pump_kw = pump_power/1000).
        pump_nom = p["p_pump_nominal"]
        pump_sigma = p.get("p_pump_sigma", 0.8)

        # Bonus gaussien: max = w['pump'] quand pump_kw == pump_nom
        pump_err = pump_kw - pump_nom
        r_pump = w["pump"] * np.exp(-0.5 * (pump_err / pump_sigma) ** 2)

        # Pénalité stabilité (seulement sur T et Flow pour ne pas figer les vannes si elles cherchent l'idéal)
        instability = np.sum(np.square(action_vector[:2]))
        r_stab = -w.get("stability", 0.0) * instability

        return float(term_comfort + term_sobriety + term_combo + term_splits + r_pump + r_stab)

    def _get_obs(self, node_temps, pipe_flows):
        demands = [self.demand_funcs[n](self.current_t) for n in self.consumer_nodes]
        obs = np.concatenate([node_temps, pipe_flows, demands], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _compute_ideal_splits(self, t):
        """
        Calcule les ratios d'ouverture idéaux basés sur la demande instantanée
        des consommateurs en aval.
        Ratio = (Demande totale branche A) / (Demande totale branche A + B)
        """
        ideal_ratios = np.zeros(len(self.branching_nodes), dtype=np.float32)
        
        for i, b_node in enumerate(self.branching_nodes):
            children = self.branching_map[b_node]
            if len(children) < 2:
                ideal_ratios[i] = 0.5
                continue
            
            # Calcul de la puissance demandée sur chaque branche
            child_1, child_2 = children[0], children[1]
            
            demand_1 = sum(self.demand_funcs[c](t) for c in self.downstream_consumers_map[b_node][child_1])
            demand_2 = sum(self.demand_funcs[c](t) for c in self.downstream_consumers_map[b_node][child_2])
            
            total_demand = demand_1 + demand_2
            
            if total_demand > 1e-3:
                # Ratio pour le premier enfant (correspond à la logique de step())
                ratio = demand_1 / total_demand
            else:
                ratio = 0.5 # Pas de demande, équilibre
                
            # On clip pour éviter de fermer totalement une vanne
            ideal_ratios[i] = np.clip(ratio, 0.05, 0.95)
            
        return ideal_ratios

    def set_agent_split_control(self, enabled: bool):
        """Méthode appelée par le Callback pour débloquer l'agent."""
        self.agent_manages_splits = enabled

    def render(self):
        print(f"T={self.current_t:.0f} | In: {self.actual_inlet_temp:.1f}C, {self.actual_mass_flow:.1f}kg/s | "
              f"Dem: {self.last_total_p_demand/1e3:.0f}kW | Sup: {self.last_total_p_supplied/1e3:.0f}kW")