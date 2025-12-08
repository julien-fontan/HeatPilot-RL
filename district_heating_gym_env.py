import gymnasium as gym
from gymnasium import spaces
import numpy as np
from district_heating_model import Pipe, DistrictHeatingNetwork
from utils import generate_step_function, generate_smooth_profile
from graph_utils import Graph
from config import (
    PHYSICAL_PROPS,
    SIMULATION_PARAMS,
    CONTROL_LIMITS,
    EDGES,
    GLOBAL_SEED,
    INITIAL_CONDITIONS,
    POWER_PROFILE_CONFIG,
    MIN_RETURN_TEMP,
    PIPE_GENERATION,   # si tu veux y accéder ici aussi
)

class HeatNetworkEnv(gym.Env):
    """
    Environnement Gym pour le contrôle du réseau de chaleur.
    """
    def __init__(self):
        super(HeatNetworkEnv, self).__init__()

        # --- Paramètres de Simulation ---
        self.dt = SIMULATION_PARAMS["dt"]       # Pas de temps de contrôle (s)
        self.t_max = SIMULATION_PARAMS["t_max_day"]     # Une journée
        self.dx = SIMULATION_PARAMS["dx"]
        # Limite dure sur le nombre de pas par épisode pour éviter des épisodes trop longs
        self.max_steps = int(2 * 3600 / self.dt)  # par ex. 2h simulées max → 2h/dt pas

        self.props = PHYSICAL_PROPS

        # --- Contraintes Physiques ---
        self.temp_min = CONTROL_LIMITS["temp_min"]
        self.temp_max = CONTROL_LIMITS["temp_max"]
        self.flow_min = CONTROL_LIMITS["flow_min"]
        self.flow_max = CONTROL_LIMITS["flow_max"]
        
        # Variation max par pas de temps
        self.max_temp_rise = CONTROL_LIMITS["max_temp_rise_per_dt"]
        self.max_temp_drop = CONTROL_LIMITS["max_temp_drop_per_dt"]
        self.max_flow_var = CONTROL_LIMITS["max_flow_delta_per_dt"]

        # --- Analyse de la Topologie ---
        self.edges = EDGES
        
        self.graph = Graph(self.edges)
        
        # Adaptation à la nouvelle API Graph
        all_children = self.graph.get_child_nodes()
        self.branching_nodes = self.graph.get_branching_nodes()
        self.branching_map = {n: all_children[n] for n in self.branching_nodes}
        
        self.inlet_node = self.graph.get_inlet_node()
        self.consumer_nodes = self.graph.get_consumer_nodes()

        # --- Espaces d'Action et d'Observation ---
        # Action: [T_inlet_target, Mass_flow_target, Split_1, Split_2, ..., Split_N]
        # Les splits sont des float entre 0..1 représentant la fraction vers le 1er enfant.
        n_actions = 2 + len(self.branching_nodes)
        self.action_space = spaces.Box(
            low=np.array([self.temp_min, self.flow_min] + [0.0]*len(self.branching_nodes), dtype=np.float32),
            high=np.array([self.temp_max, self.flow_max] + [1.0]*len(self.branching_nodes), dtype=np.float32),
            dtype=np.float32
        )

        # Observation: 
        # - Température actuelle aux noeuds consommateurs (len(consumer_nodes))
        # - Température actuelle à l'entrée (1)
        # - Débit actuel (1)
        # - Demande de puissance actuelle pour chaque consommateur (len(consumer_nodes))
        n_obs = len(self.consumer_nodes) + 1 + 1 + len(self.consumer_nodes)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32)

        self.network = None
        self.state = None
        self.current_t = 0.0
        self.demand_funcs = {}  # node_id -> f(t)
        
        self.actual_inlet_temp = INITIAL_CONDITIONS["initial_temp"]
        self.actual_mass_flow = INITIAL_CONDITIONS["initial_flow"]

        # suivi des puissances au dernier pas
        self.last_total_p_demand = 0.0
        self.last_total_p_supplied = 0.0
        self.last_p_source = 0.0
        self.last_p_pump = 0.0

        # compteur de pas dans l'épisode courant
        self.step_count = 0

    def reset(self, seed=None, options=None):
        """
        Reset conforme à l'API Gymnasium:
        - accepte seed et options,
        - retourne (observation, info).
        """
        # Si aucun seed n'est fourni, on utilise GLOBAL_SEED pour la reproductibilité
        if seed is None:
            seed = GLOBAL_SEED

        super().reset(seed=seed)

        edges = self.edges

        # 1. Génération reproductible des propriétés des conduites, y compris heat_loss_coeff
        lengths, diameters, n_segments, h_vals = Pipe.generate_parameters(
            edges=edges,
            dx=self.dx,
            seed=seed,
            # les bornes viennent de PIPE_GENERATION par défaut
        )

        pipes_list = []
        for i, (u_node, v_node) in enumerate(edges):
            p = Pipe(
                nodes=(u_node, v_node),
                length=float(lengths[i]),
                diameter=float(diameters[i]),
                dx=self.dx,
                rho=self.props["rho"],
                cp=self.props["cp"],
                heat_loss_coeff=float(h_vals[i]),
                thermal_conductivity=self.props["thermal_conductivity"],
                external_temp=self.props["external_temp"],
            )
            pipes_list.append(p)

        # 2. Génération des profils de demande (puissance en W) pour les consommateurs
        # step_time est un temps physique (s). smooth_factor > 1.0 rend le profil plus lisse.
        rng = np.random.default_rng()  # profils différents à chaque reset
        smooth_factor = POWER_PROFILE_CONFIG.get("smooth_factor", 1.0)
        self.demand_funcs = {}
        for node in self.consumer_nodes:
            self.demand_funcs[node] = generate_smooth_profile(
                t_end=self.t_max,
                step_time=POWER_PROFILE_CONFIG["step_time"],
                min_val=POWER_PROFILE_CONFIG["p_min"],
                max_val=POWER_PROFILE_CONFIG["p_max"],
                seed=rng.integers(0, 1_000_000),
                smooth_factor=smooth_factor,
            )

        # 3. Construire le réseau avec consommation activée
        self.network = DistrictHeatingNetwork(
            pipes=pipes_list,
            inlet_mass_flow=INITIAL_CONDITIONS["initial_flow"],
            node_splits={},
            graph=self.graph,
            inlet_temp=INITIAL_CONDITIONS["initial_temp"],
            rho=self.props["rho"],
            cp=self.props["cp"],
            node_power_funcs=self.demand_funcs,
            t_min_return=MIN_RETURN_TEMP,
        )

        # 4. Initialisation état
        self.state = np.full(self.network.total_cells, INITIAL_CONDITIONS["initial_temp"])  # °C partout
        self.current_t = 0.0
        self.actual_inlet_temp = INITIAL_CONDITIONS["initial_temp"]
        self.actual_mass_flow = INITIAL_CONDITIONS["initial_flow"]
        
        self.last_total_p_demand = 0.0
        self.last_total_p_supplied = 0.0
        self.last_p_source = 0.0
        self.last_p_pump = 0.0

        # remise à zéro du compteur de pas
        self.step_count = 0

        return self._get_obs(), {}

    def step(self, action):
        # Déballe l'action
        target_temp = float(action[0])
        target_flow = float(action[1])
        split_actions = action[2:]

        # --- Application des contraintes (rampes de températures) ---
        
        # 1. Température (+0.5 max, ou baisse rapide)
        delta_temp = target_temp - self.actual_inlet_temp
        if delta_temp > self.max_temp_rise:
            self.actual_inlet_temp += self.max_temp_rise
        elif delta_temp < -self.max_temp_drop:
            self.actual_inlet_temp -= self.max_temp_drop
        else:
            self.actual_inlet_temp = target_temp
            
        # Vérifie que la température reste dans les bornes
        self.actual_inlet_temp = np.clip(self.actual_inlet_temp, self.temp_min, self.temp_max)

        # 2. Débit massique (rampes encore une fois)
        delta_flow = target_flow - self.actual_mass_flow
        delta_flow = np.clip(delta_flow, -self.max_flow_var, self.max_flow_var)
        self.actual_mass_flow += delta_flow
        self.actual_mass_flow = np.clip(self.actual_mass_flow, self.flow_min, self.flow_max)

        # 3. Fractions de répartition du débit aux noeuds ramifiés
        # Conversion des actions (0..1) en dictionnaire pour le réseau
        node_splits = {}
        for i, node in enumerate(self.branching_nodes):
            frac_first = np.clip(split_actions[i], 0.01, 0.99) # Eviter 0 ou 1 strict
            children = self.branching_map[node]
            
            # Important: On suppose ici des branchements binaires (1 entrée -> 2 sorties)
            # Si > 2 sorties, il faudrait plus d'actions par noeud.
            if len(children) >= 2:
                node_splits[node] = {
                    children[0]: frac_first,
                    children[1]: 1.0 - frac_first
                }
            # Si plus de 2 enfants, les autres auront 0 avec cette logique simple, ou il faut adapter l'espace d'action.

        # --- Mise à jour du réseau ---
        self.network.inlet_mass_flow = self.actual_mass_flow
        self.network.inlet_temp = self.actual_inlet_temp  # maj explicite de T_in
        self.network.set_node_splits(node_splits)
        
        # --- Simulation (1 pas de temps) ---
        t_next = self.current_t + self.dt
        sol = self.network.solve(
            (self.current_t, t_next),
            self.state,
            method="RK45",
            rtol=SIMULATION_PARAMS["rtol"],
            atol=SIMULATION_PARAMS["atol"],
        )
        self.state = sol.y[:, -1]
        self.current_t = t_next
        self.step_count += 1

        # Après ce pas, DistrictHeatingNetwork a:
        # - recalculé mass_flows (dans compute_system_dynamics),
        # - appliqué les puissances (si enable_consumption=True),
        # - stocké node_temps corrigés dans last_node_temps.
        mass_flows = self.network._compute_mass_flows(self.current_t)
        node_temps = self.network.last_node_temps

        # --- Calcul des puissances réellement fournies aux consommateurs ---
        cp = self.props["cp"]
        parents_map = self.graph.get_parent_nodes()
        total_mismatch = 0.0
        total_p_demand = 0.0
        total_p_supplied = 0.0

        for node_id in self.consumer_nodes:
            # demande cible
            p_target = self.demand_funcs[node_id](self.current_t)
            total_p_demand += p_target

            # débit entrant
            parents = parents_map.get(node_id, [])
            if not parents:
                continue

            m_in = 0.0
            for parent_node in parents:
                edge_data = self.graph.edges[(parent_node, node_id)]
                pipe_idx = edge_data["pipe_index"]
                m_in += mass_flows[pipe_idx]

            if m_in <= 0.0:
                # pas de débit → aucune puissance réellement soutirée
                p_supplied = 0.0
            else:
                # T_in et T_out au noeud
                node_idx = self.network.graph.get_id_from_node(node_id)
                # T_out = température nodale après consommation (node_temps)
                T_out = node_temps[node_idx]
                # p_supplied ≈ min(p_target, m_in * cp * (T_in - t_min_return)).
                p_supplied = min(
                    p_target,
                    m_in * cp * max(T_out - MIN_RETURN_TEMP, 0.0)
                )

            total_p_supplied += p_supplied
            total_mismatch += abs(p_supplied - p_target)

        # --- Coût à la source ---
        p_source = self.actual_mass_flow * cp * (self.actual_inlet_temp - MIN_RETURN_TEMP)
        p_pump = 1000.0 * self.actual_mass_flow  # pas de modèle hydraulique, on prend un deltaP fixé arbitraire

        reward = - (1.0e-4 * total_mismatch + 1.0e-5 * p_source + 1.0e-2 * p_pump)

        # mémoriser pour render()
        self.last_total_p_demand = total_p_demand
        self.last_total_p_supplied = total_p_supplied
        self.last_p_source = p_source
        self.last_p_pump = p_pump

        # --- Observation ---
        obs = self._get_obs()
        
        terminated = False
        # Troncature si temps max atteint OU nombre de pas max atteint
        truncated = (self.current_t >= self.t_max) or (self.step_count >= self.max_steps)
        
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        # Récupérer températures aux noeuds consommateurs
        temps = []
        parents_map = self.graph.get_parent_nodes()
        
        for node_id in self.consumer_nodes:
            parents = parents_map.get(node_id, [])
            if parents:
                parent_node = parents[0]
                edge_data = self.graph.edges[(parent_node, node_id)]
                pipe_idx = edge_data["pipe_index"]
                
                sl = self.network.pipe_slices[pipe_idx]
                temps.append(self.state[sl.stop - 1])
            else:
                temps.append(20.0) # Défaut

        demands = [self.demand_funcs[n](self.current_t) for n in self.consumer_nodes]
        
        obs = np.array(
            temps + 
            [self.actual_inlet_temp, self.actual_mass_flow] + 
            demands, 
            dtype=np.float32
        )
        return obs

    def render(self):
        print(
            f"Time: {self.current_t:.1f}s | "
            f"Temperature in: {self.actual_inlet_temp:.2f}°C | "
            f"Mass flow in: {self.actual_mass_flow:.2f} kg/s | "
            f"P_demand_tot: {self.last_total_p_demand/1000:.1f} kW | "
            f"P_supplied_tot: {self.last_total_p_supplied/1000:.1f} kW | "
            f"P_source+pump: {(self.last_p_source + self.last_p_pump)/1000:.1f} kW"
        )
