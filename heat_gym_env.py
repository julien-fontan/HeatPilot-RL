import gymnasium as gym
from gymnasium import spaces
import numpy as np
from heat_network import Pipe, DistrictHeatingNetwork
from utils import generate_step_function, PhysicalProperties
from graph_utils import Graph

class HeatNetworkEnv(gym.Env):
    """
    Environnement Gym pour le contrôle du réseau de chaleur.
    """
    def __init__(self, edges):
        super(HeatNetworkEnv, self).__init__()

        # --- Paramètres de Simulation ---
        self.dt = 10.0  # Pas de temps de contrôle (s)
        self.t_max = 24 * 3600.0  # Une journée
        self.dx = 10.0
        
        self.props = PhysicalProperties(
            rho=1000.0,
            cp=4182.0,
            thermal_conductivity=0.0,
            external_temp=10.0,
            heat_loss_coeff=1.5
        )
        
        # --- Contraintes Physiques ---
        self.temp_min = 60.0
        self.temp_max = 110.0
        self.flow_min = 1.0
        self.flow_max = 20.0
        
        # Variation max par pas de temps (10s)
        self.max_temp_rise = 0.5  # +0.5°C / 10s
        self.max_temp_drop = 5.0  # Supposons une baisse rapide possible (ex: 5°C/10s)
        self.max_flow_var = 1.0   # kg/s / 10s

        # --- Analyse de la Topologie ---
        self.edges = edges
        
        # Utilisation de la bibliothèque graph_utils pour l'analyse
        # On stocke le graphe dans self.graph pour le réutiliser
        self.graph = Graph(self.edges)
        
        self.branching_map = self.graph.get_branching_map()
        self.inlet_node, self.consumer_nodes, self.branching_nodes = self.graph.get_special_nodes()

        print(f"Topologie détectée : {self.graph.get_nodes_count()} noeuds, {len(self.edges)} tuyaux.")
        print(f"Entrée : {self.inlet_node}")
        print(f"Consommateurs : {self.consumer_nodes}")
        print(f"Branchements : {self.branching_nodes}")

        # --- Espaces d'Action et d'Observation ---
        # Action: [T_inlet_target, Mass_flow_target, Split_1, Split_2, ..., Split_N]
        # Les splits sont des valeurs continues 0..1 représentant la fraction vers le 1er enfant.
        n_actions = 2 + len(self.branching_nodes)
        self.action_space = spaces.Box(
            low=np.array([self.temp_min, self.flow_min] + [0.0]*len(self.branching_nodes)),
            high=np.array([self.temp_max, self.flow_max] + [1.0]*len(self.branching_nodes)),
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
        self.demand_funcs = {}
        
        # État interne des actionneurs (pour gérer les rampes)
        self.actual_inlet_temp = 70.0
        self.actual_mass_flow = 10.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)

        # 1. Reconstruire le réseau (pour reset l'état thermique interne)
        pipes_list = []
        # Génération aléatoire des propriétés (comme dans main.py)
        n_pipes = len(self.edges)
        n_segments = rng.integers(20, 61, size=n_pipes)
        lengths = self.dx * n_segments
        diameters = rng.uniform(0.15, 0.35, size=n_pipes)

        for i, (u_node, v_node) in enumerate(self.edges):
            p = Pipe(
                nodes=(u_node, v_node), length=float(lengths[i]),
                diameter=float(diameters[i]), 
                dx=self.dx, props=self.props
            )
            pipes_list.append(p)

        # On passe self.graph pour éviter de recalculer la topologie
        self.network = DistrictHeatingNetwork(
            pipes=pipes_list, props=self.props,
            inlet_node_id=self.inlet_node, inlet_mass_flow=10.0, node_splits={},
            graph=self.graph
        )

        # 2. Générer les profils de demande (Puissance en Watts)
        self.demand_funcs = {}
        for node in self.consumer_nodes:
            # Demande entre 10kW et 200kW
            self.demand_funcs[node] = generate_step_function(
                self.t_max, 3600.0, 10000.0, 200000.0, seed=rng.integers(0, 10000)
            )

        # 3. Initialisation état
        self.state = np.full(self.network.total_cells, 70.0)
        self.current_t = 0.0
        self.actual_inlet_temp = 70.0
        self.actual_mass_flow = 10.0
        
        return self._get_obs(), {}

    def step(self, action):
        # Déballage de l'action
        target_temp = float(action[0])
        target_flow = float(action[1])
        split_actions = action[2:]

        # --- Application des Contraintes (Rampes) ---
        
        # 1. Température (+0.5 max, ou baisse rapide)
        delta_temp = target_temp - self.actual_inlet_temp
        if delta_temp > self.max_temp_rise:
            self.actual_inlet_temp += self.max_temp_rise
        elif delta_temp < -self.max_temp_drop:
            self.actual_inlet_temp -= self.max_temp_drop
        else:
            self.actual_inlet_temp = target_temp
            
        # Clip aux bornes absolues
        self.actual_inlet_temp = np.clip(self.actual_inlet_temp, self.temp_min, self.temp_max)

        # 2. Débit massique (Variation bornée)
        delta_flow = target_flow - self.actual_mass_flow
        delta_flow = np.clip(delta_flow, -self.max_flow_var, self.max_flow_var)
        self.actual_mass_flow += delta_flow
        self.actual_mass_flow = np.clip(self.actual_mass_flow, self.flow_min, self.flow_max)

        # 3. Splits
        # Conversion des actions (0..1) en dictionnaire pour le réseau
        node_splits = {}
        for i, node in enumerate(self.branching_nodes):
            frac_first = np.clip(split_actions[i], 0.01, 0.99) # Eviter 0 ou 1 strict
            children = self.branching_map[node]
            
            # Note: On suppose ici des branchements binaires (1 entrée -> 2 sorties)
            # Si > 2 sorties, il faudrait plus d'actions par noeud.
            if len(children) >= 2:
                node_splits[node] = {
                    children[0]: frac_first,
                    children[1]: 1.0 - frac_first
                }
            # Si plus de 2 enfants, les autres auront 0 avec cette logique simple, 
            # ou il faut adapter l'espace d'action.

        # --- Mise à jour du Réseau ---
        self.network.inlet_mass_flow = self.actual_mass_flow
        self.network.set_node_splits(node_splits)
        
        # Conditions aux limites pour ce pas de temps
        boundary_conditions = {self.inlet_node: self.actual_inlet_temp}

        # --- Simulation (1 pas de temps) ---
        t_next = self.current_t + self.dt
        sol = self.network.solve(
            (self.current_t, t_next),
            self.state,
            boundary_conditions,
            method="RK45", # RK45 est souvent plus rapide pour des petits pas que BDF
            rtol=1e-3, atol=1e-3
        )
        
        # Mise à jour de l'état interne
        self.state = sol.y[:, -1]
        self.current_t = t_next

        # --- Calcul des Récompenses ---
        # 1. Puissance fournie vs Demandée
        # On estime la puissance reçue par P = m_dot * cp * (T_node - T_ref). 
        # T_ref est arbitraire (ex: 40°C retour), ici on simplifie en supposant que la demande est une puissance thermique nette.
        
        mass_flows = self.network._compute_mass_flows(self.current_t) # Débits actuels dans les tuyaux
        
        # Mapping pipe -> node aval pour savoir quel débit arrive au consommateur
        # On cherche le tuyau qui arrive au noeud consommateur
        total_mismatch = 0.0
        
        # Récupération des températures aux noeuds (approximation via la fin des tuyaux entrants)
        # Pour simplifier, on prend la température moyenne du noeud calculée par le réseau s'il y avait mélange,
        # mais ici on peut prendre la température de la dernière cellule du tuyau entrant.
        
        node_temps = np.zeros(self.network.n_nodes)
        # On reconstruit vite fait les températures aux noeuds
        # (Note: Idéalement network devrait exposer ça publiquement après solve)
        # On utilise une heuristique simple: T_noeud ~ T_fin_tuyau_entrant
        
        for node_id in self.consumer_nodes:
            node_idx = self.network.node_map[node_id]
            in_pipes = self.network.node_in_pipes[node_idx]
            if not in_pipes: continue
            
            # On suppose un seul tuyau entrant pour les terminaux dans cette topologie
            pipe_idx = in_pipes[0] 
            m_dot = mass_flows[pipe_idx]
            
            # Température à la fin du tuyau
            sl = self.network.pipe_slices[pipe_idx]
            t_fluid = self.state[sl.stop - 1]
            
            # Puissance fournie (ref 40°C pour le retour fictif)
            p_supplied = m_dot * self.cp * (t_fluid - 40.0)
            if p_supplied < 0: p_supplied = 0
            
            p_target = self.demand_funcs[node_id](self.current_t)
            
            # Pénalité quadratique ou absolue sur l'écart
            total_mismatch += abs(p_supplied - p_target)

        # 2. Coût énergétique (Minimiser T_in et m_in)
        # Puissance injectée à la source
        p_source = self.actual_mass_flow * self.cp * (self.actual_inlet_temp - 40.0)
        # Puissance pompage (proportionnelle au débit ou cube du débit)
        p_pump = 1000.0 * self.actual_mass_flow # Facteur arbitraire pour pondérer
        
        # Reward = - (Poids * Mismatch + Poids * Coût)
        reward = - (1.0e-4 * total_mismatch + 1.0e-5 * p_source + 1.0e-2 * p_pump)

        # --- Observation ---
        obs = self._get_obs()
        
        terminated = False
        truncated = (self.current_t >= self.t_max)
        
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        # Récupérer températures aux noeuds consommateurs
        temps = []
        # On doit mapper node_id -> index cellule fin de tuyau entrant
        # C'est un peu lourd, on va faire une approx rapide ou stocker les indices au reset
        # Pour l'instant, on refait la logique simple
        for node_id in self.consumer_nodes:
            node_idx = self.network.node_map[node_id]
            in_pipes = self.network.node_in_pipes[node_idx]
            if in_pipes:
                pipe_idx = in_pipes[0]
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
        print(f"Time: {self.current_t:.1f}s | T_in: {self.actual_inlet_temp:.2f} | Flow: {self.actual_mass_flow:.2f}")
