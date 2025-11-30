import numpy as np
from scipy.integrate import solve_ivp

class Pipe:
    """
    Représente une conduite unique.
    Optimisation : Utilisation de __slots__ et pré-calcul des constantes.
    """
    __slots__ = ('id', 'nodes', 'length', 'diameter', 'dx', 'n_cells', 
                 'area', 'loss_factor', 'diffusivity_factor', 'external_temp')

    def __init__(self, id, nodes, length, diameter, heat_loss_coeff, dx, 
                 external_temp=10.0, thermal_conductivity=0.0, rho=1000.0, cp=4182.0):
        self.id = id
        self.nodes = nodes
        self.length = length
        self.diameter = diameter
        self.dx = dx
        self.external_temp = external_temp
        
        self.n_cells = int(round(length / dx))
        if self.n_cells < 2:
            raise ValueError(f"Conduite {id}: Longueur insuffisante pour le pas dx choisi.")

        # --- Pré-calculs des constantes physiques ---
        # Aire de la section (m²)
        self.area = np.pi * (diameter / 2)**2
        
        # Facteur de perte (1/s) : constant si rho/cp constants
        self.loss_factor = (4.0 * heat_loss_coeff) / (diameter * rho * cp)
        
        # Facteur de diffusion (1/s) : alpha / dx^2
        diffusivity = thermal_conductivity / (rho * cp) if thermal_conductivity > 0 else 0
        self.diffusivity_factor = diffusivity / (dx**2) if diffusivity > 0 else 0.0

    def compute_derivatives_inplace(self, t, T, velocity, inlet_temp, out_view):
        """
        Calcule dT/dt et écrit directement dans le tableau 'out_view' (pas d'allocation).
        """
        # Récupération rapide
        dx = self.dx
        loss = self.loss_factor
        T_ext = self.external_temp(t) if callable(self.external_temp) else self.external_temp
        
        # Advection : Upwind scheme
        # Si v > 0 : flux de 0 vers N. Amont = i-1
        # Si v < 0 : flux de N vers 0. Amont = i+1
        
        u_dx = velocity / dx
        
        if velocity >= 0:
            # Flux positif (Forward)
            # Cellule 0 (Entrée)
            out_view[0] = -u_dx * (T[0] - inlet_temp) - loss * (T[0] - T_ext)
            # Cellules internes (1 à N-1)
            # dT/dt = -u * (T[i] - T[i-1])/dx
            out_view[1:] = -u_dx * (T[1:] - T[:-1]) - loss * (T[1:] - T_ext)
        else:
            # Flux négatif (Backward)
            # Cellules internes (0 à N-2)
            # dT/dt = -u * (T[i] - T[i+1])/dx  (attention u est négatif, donc -u > 0)
            out_view[:-1] = -u_dx * (T[:-1] - T[1:]) - loss * (T[:-1] - T_ext)
            # Cellule N-1 (Entrée du flux inverse)
            out_view[-1] = -u_dx * (T[-1] - inlet_temp) - loss * (T[-1] - T_ext)

        # Diffusion (Optionnel, souvent nul)
        if self.diffusivity_factor > 0:
            k = self.diffusivity_factor
            # Laplacien discret 1D
            diffusion = np.zeros_like(T)
            diffusion[1:-1] = k * (T[:-2] - 2*T[1:-1] + T[2:])
            # Conditions aux bords simplifiées pour la diffusion (Neumann ou Dirichlet implicite)
            diffusion[0] = k * (T[1] - T[0])
            diffusion[-1] = k * (T[-2] - T[-1])
            out_view += diffusion


class DistrictHeatingNetwork:
    def __init__(self, pipes, rho=1000.0, cp=4182.0):
        self.pipes = pipes
        self.rho = rho
        self.cp = cp
        self._build_topology()

    def _build_topology(self):
        """
        Construit les structures de données vectorisées pour le graphe.
        """
        # 1. Mappage des IDs de noeuds vers des entiers 0..N-1
        unique_nodes = set()
        for p in self.pipes:
            unique_nodes.add(p.nodes[0])
            unique_nodes.add(p.nodes[1])
        
        self.node_map = {id: i for i, id in enumerate(sorted(list(unique_nodes)))}
        self.n_nodes = len(self.node_map)
        self.inv_node_map = {v: k for k, v in self.node_map.items()}

        # 2. Structures vectorielles pour les tuyaux
        n_pipes = len(self.pipes)
        self.pipe_u_indices = np.zeros(n_pipes, dtype=int) # Index noeud amont
        self.pipe_v_indices = np.zeros(n_pipes, dtype=int) # Index noeud aval
        self.pipe_areas = np.zeros(n_pipes)
        
        # 3. Gestion du vecteur d'état global
        self.pipe_slices = []
        offset = 0
        
        # Indices pour récupérer rapidement les températures aux extrémités des tuyaux
        self.idx_pipe_start = [] 
        self.idx_pipe_end = []

        for i, p in enumerate(self.pipes):
            u, v = p.nodes
            self.pipe_u_indices[i] = self.node_map[u]
            self.pipe_v_indices[i] = self.node_map[v]
            self.pipe_areas[i] = p.area
            
            sl = slice(offset, offset + p.n_cells)
            self.pipe_slices.append(sl)
            
            self.idx_pipe_start.append(offset)
            self.idx_pipe_end.append(offset + p.n_cells - 1)
            
            offset += p.n_cells
            
        self.total_cells = offset
        self.idx_pipe_start = np.array(self.idx_pipe_start)
        self.idx_pipe_end = np.array(self.idx_pipe_end)

    def compute_system_dynamics(self, t, global_state, velocity_funcs, boundary_conditions):
        """
        Version optimisée : vectorisation des mélanges et allocations réduites.
        """
        dy_dt = np.empty_like(global_state)
        
        # 1. Récupération des vitesses (Vectorisation difficile car fonctions différentes, mais liste comp rapide)
        # On suppose que velocity_funcs est une liste de callables ou de floats
        velocities = np.array([f(t) if callable(f) else f for f in velocity_funcs])
        
        # 2. Calcul vectorisé des débits massiques
        # m_dot = rho * Area * v
        mass_flows = self.rho * self.pipe_areas * velocities
        
        # 3. Résolution des températures aux noeuds (Mélange vectorisé)
        node_temps = self._solve_node_mixing(global_state, mass_flows, boundary_conditions, t)
        
        # 4. Calcul de la physique par conduite
        # On itère encore sur les objets Pipe car ils ont des géométries (dx) différentes,
        # mais on utilise la méthode 'inplace' pour éviter les allocations.
        for i, p in enumerate(self.pipes):
            u_idx = self.pipe_u_indices[i]
            v_idx = self.pipe_v_indices[i]
            m_dot = mass_flows[i]
            
            # Température entrante selon le sens du flux
            temp_inlet = node_temps[u_idx] if m_dot >= 0 else node_temps[v_idx]
            
            # Vue sur la partie du vecteur d'état concernée
            sl = self.pipe_slices[i]
            p.compute_derivatives_inplace(t, global_state[sl], velocities[i], temp_inlet, dy_dt[sl])
            
        return dy_dt

    def _solve_node_mixing(self, global_state, mass_flows, boundary_conditions, t):
        """
        Calcule la température de mélange aux noeuds sans boucles Python lentes.
        Utilise np.add.at pour accumuler les flux.
        """
        # Initialisation des accumulateurs
        mass_in = np.zeros(self.n_nodes)
        heat_in = np.zeros(self.n_nodes)
        
        # Masques booléens pour le sens du flux
        flow_pos = mass_flows > 0
        flow_neg = ~flow_pos
        
        # --- Flux Positifs (u -> v) ---
        # Le noeud v reçoit de la masse et de la chaleur venant de la fin du tuyau
        target_nodes_pos = self.pipe_v_indices[flow_pos]
        mass_vals_pos = mass_flows[flow_pos] # Déjà positifs
        # Température sortant du tuyau (fin)
        temp_vals_pos = global_state[self.idx_pipe_end[flow_pos]]
        
        np.add.at(mass_in, target_nodes_pos, mass_vals_pos)
        np.add.at(heat_in, target_nodes_pos, mass_vals_pos * temp_vals_pos)
        
        # --- Flux Négatifs (v -> u) ---
        # Le noeud u reçoit de la masse et de la chaleur venant du début du tuyau
        target_nodes_neg = self.pipe_u_indices[flow_neg]
        mass_vals_neg = -mass_flows[flow_neg] # On veut la valeur absolue pour la masse
        # Température sortant du tuyau (début, car flux inverse)
        temp_vals_neg = global_state[self.idx_pipe_start[flow_neg]]
        
        np.add.at(mass_in, target_nodes_neg, mass_vals_neg)
        np.add.at(heat_in, target_nodes_neg, mass_vals_neg * temp_vals_neg)
        
        # --- Calcul final (Moyenne pondérée) ---
        # Évite la division par zéro
        with np.errstate(divide='ignore', invalid='ignore'):
            node_temps = heat_in / mass_in
            
        # Gestion des noeuds isolés (mass_in == 0) -> Température par défaut ou maintien
        node_temps[np.isnan(node_temps)] = 10.0 

        # --- Application des conditions aux limites ---
        # On écrase les valeurs calculées par les BCs imposées
        for node_id, func in boundary_conditions.items():
            if node_id in self.node_map:
                idx = self.node_map[node_id]
                val = func(t) if callable(func) else func
                node_temps[idx] = val
                
        return node_temps

    def solve(self, tspan, initial_state, velocity_funcs, boundary_conditions, **kwargs):
        fun = lambda t, y: self.compute_system_dynamics(t, y, velocity_funcs, boundary_conditions)
        return solve_ivp(fun, tspan, initial_state, **kwargs)
