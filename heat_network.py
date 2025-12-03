import numpy as np
from scipy.integrate import solve_ivp
from graph_utils import Graph

class Pipe:

    def __init__(self, nodes, length, diameter, dx, props):
        self.props = props
        self.nodes = nodes
        self.area = np.pi * (diameter / 2)**2
        self.loss_factor = (4.0 * props.heat_loss_coeff) / (diameter * props.rho * props.cp)  # Facteur de perte (1/s)
        self.diffusivity = props.thermal_conductivity / (props.rho * props.cp) if props.thermal_conductivity > 0 else 0  # Facteur de diffusion (1/s) : alpha / dx^2
        self.dx = dx
        self.n_cells = int(round(length / dx))
        if self.n_cells < 2:
            raise ValueError(f"Conduite {nodes}: Longueur insuffisante pour le pas dx choisi.")


    def compute_derivatives_inplace(self, t, T, mass_flow, inlet_temp, dT_dt):
        """
        Calcule dT/dt
        """
        ext_temp = self.props.external_temp
        T_ext = ext_temp(t) if callable(ext_temp) else ext_temp
        rho = self.props.rho

        if mass_flow >= 0:  # Flux positif (Forward)
            # Cellule 0 (Entrée)
            dT_dt[0] = -(mass_flow / (self.dx * rho * self.area)) * (T[0] - inlet_temp) - self.loss_factor * (T[0] - T_ext)
            # Cellules internes (1 à N-1)
            # dT/dt = -u * (T[i] - T[i-1])/dx
            dT_dt[1:] = -(mass_flow / (self.dx * rho * self.area)) * (T[1:] - T[:-1]) - self.loss_factor * (T[1:] - T_ext)
        else:   # Flux négatif (Backward)
            # Cellules internes (0 à N-2)
            # dT/dt = -u * (T[i] - T[i+1])/dx  (attention u est négatif, donc -u > 0)
            dT_dt[:-1] = -(mass_flow / (self.dx * rho * self.area)) * (T[:-1] - T[1:]) - self.loss_factor * (T[:-1] - T_ext)
            # Cellule N-1 (Entrée du flux inverse)
            dT_dt[-1] = -(mass_flow / (self.dx * rho * self.area)) * (T[-1] - inlet_temp) - self.loss_factor * (T[-1] - T_ext)

        # Diffusion (Optionnel, souvent nul)
        if self.diffusivity > 0:
            k = self.diffusivity/(self.dx**2)
            # Laplacien discret 1D
            diffusion = np.zeros_like(T)
            diffusion[1:-1] = k * (T[:-2] - 2*T[1:-1] + T[2:])
            # Conditions aux bords simplifiées pour la diffusion (Neumann ou Dirichlet implicite)
            diffusion[0] = k * (T[1] - T[0])
            diffusion[-1] = k * (T[-2] - T[-1])
            dT_dt += diffusion


class DistrictHeatingNetwork:
    def __init__(self, graph, pipes, props,
                 inlet_node_id=None, inlet_mass_flow=0.0, node_splits=None):
        
        self.graph = graph
        self.pipes = pipes
        self.props = props
        
        self.node_map, self.inv_node_map = self.graph.get_node_maps()
        self.n_nodes = self.graph.get_nodes_count()
        (self.node_in_pipes, self.node_out_pipes, 
         self.pipe_u_indices, self.pipe_v_indices) = self.graph.get_adjacency_indices()
        self.topo_nodes = self.graph.get_topological_sort()

        # 2. Initialisation des slices (spécifique à la physique)
        self._init_pipe_slices()

        # Définition noeud d'entrée
        if inlet_node_id is None:
            self.inlet_node_id, _, _ = self.graph.get_special_nodes()
        else:
            self.inlet_node_id = inlet_node_id

        # Débit massique d'entrée (float ou callable(t))
        self.inlet_mass_flow = inlet_mass_flow

        # Dictionnaire : node_id -> {next_node_id: fraction}
        if node_splits is None:
            node_splits = {}
        self._build_split_fractions(node_splits)

    def _init_pipe_slices(self):
        self.pipe_areas = np.zeros(len(self.pipes))
        self.pipe_slices = []
        offset = 0
        self.idx_pipe_start = [] 
        self.idx_pipe_end = []

        for i, p in enumerate(self.pipes):
            self.pipe_areas[i] = p.area
            sl = slice(offset, offset + p.n_cells)
            self.pipe_slices.append(sl)
            self.idx_pipe_start.append(offset)
            self.idx_pipe_end.append(offset + p.n_cells - 1)
            offset += p.n_cells

        self.total_cells = offset
        self.idx_pipe_start = np.array(self.idx_pipe_start)
        self.idx_pipe_end = np.array(self.idx_pipe_end)

    def _build_split_fractions(self, node_splits):
        """
        Tableau 1D pipe_split_fraction[i] : fraction de débit allouée à la conduite i.
        """
        n_pipes = len(self.pipes)
        self.pipe_split_fraction = np.zeros(n_pipes, dtype=float)

        # Pour chaque noeud, on récupère ses pipes sortantes et on remplit les fractions
        for n in range(self.n_nodes):
            out_pipes = self.node_out_pipes[n]
            if not out_pipes:
                continue

            logical_node_id = self.inv_node_map[n]
            splits = node_splits.get(logical_node_id, None)

            if splits is None:
                # Uniforme si aucune règle n'est donnée
                frac = np.full(len(out_pipes), 1.0 / len(out_pipes))
            else:
                frac = np.zeros(len(out_pipes), dtype=float)
                for j, pipe_idx in enumerate(out_pipes):
                    v_idx = self.pipe_v_indices[pipe_idx]
                    v_logical = self.inv_node_map[v_idx]
                    frac[j] = splits.get(v_logical, 0.0)
                s = frac.sum()
                if s > 0:
                    frac /= s
                else:
                    frac[:] = 1.0 / len(out_pipes)

            # Stockage dans le tableau global par pipe
            for j, pipe_idx in enumerate(out_pipes):
                self.pipe_split_fraction[pipe_idx] = frac[j]

    def set_node_splits(self, node_splits):
        """
        Met à jour les fractions de répartition dynamiquement.
        """
        self._build_split_fractions(node_splits)

    def _compute_mass_flows(self, t):
        """
        Débits massiques dans chaque conduite à partir :
        - du noeud d'entrée self.inlet_node_id
        - du débit massique d'entrée self.inlet_mass_flow
        - des fractions self.pipe_split_fraction
        """
        n_pipes = len(self.pipes)
        mass_flows = np.zeros(n_pipes, dtype=float)
        node_mass_in = np.zeros(self.n_nodes, dtype=float)

        inlet_node = self.node_map[self.inlet_node_id]
        if callable(self.inlet_mass_flow):
            inlet_mass = float(self.inlet_mass_flow(t))
        else:
            inlet_mass = float(self.inlet_mass_flow)
        node_mass_in[inlet_node] += inlet_mass

        for n in self.topo_nodes:
            incoming = node_mass_in[n]
            if incoming <= 0.0:
                continue
            out_pipes = self.node_out_pipes[n]
            if not out_pipes:
                continue

            # Répartition du débit massique entrant en fonction des fractions de chaque conduite sortante
            for pipe_idx in out_pipes:
                frac = self.pipe_split_fraction[pipe_idx]
                m_dot = incoming * frac
                mass_flows[pipe_idx] = m_dot
                v = self.pipe_v_indices[pipe_idx]
                node_mass_in[v] += m_dot

        return mass_flows

    def compute_system_dynamics(self, t, global_state, boundary_conditions):

        dy_dt = np.empty_like(global_state)

        # 1. Débits massiques
        mass_flows = self._compute_mass_flows(t)

        # 2. Mélange aux noeuds
        node_temps = self._solve_node_mixing(global_state, mass_flows, boundary_conditions, t)
        
        # Récupération indices pour la boucle physique
        _, _, pipe_u_indices, pipe_v_indices = self.graph.get_adjacency_indices()

        # 3. Physique par conduite
        for i, p in enumerate(self.pipes):
            u_idx = pipe_u_indices[i]
            v_idx = pipe_v_indices[i]
            m_dot = mass_flows[i]

            temp_inlet = node_temps[u_idx] if m_dot >= 0 else node_temps[v_idx]
            sl = self.pipe_slices[i]
            p.compute_derivatives_inplace(t, global_state[sl], m_dot, temp_inlet, dy_dt[sl])

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

    def solve(self, tspan, initial_state, boundary_conditions, **kwargs):
        fun = lambda t, y: self.compute_system_dynamics(t, y, boundary_conditions)
        return solve_ivp(fun, tspan, initial_state, **kwargs)    
    
    def get_pipes_temperature(self, solution_y):
        """
        Récupère les températures moyennes de chaque pipe pour chaque pas de temps.
        
        Args:
            solution_y: La matrice solution retournée par solve_ivp (shape: (total_cells, n_time_steps))
            
        Returns:
            np.ndarray: Matrice de forme (n_pipes, n_time_steps) contenant la température moyenne de chaque pipe.
        """
        n_time_steps = solution_y.shape[1]
        n_pipes = len(self.pipes)
        temp_matrix = np.zeros((n_pipes, n_time_steps))
        
        for i, sl in enumerate(self.pipe_slices):
            # On prend la moyenne spatiale de la température dans le tuyau à chaque instant
            temp_matrix[i, :] = np.mean(solution_y[sl, :], axis=0)
            
        return temp_matrix

    def get_flows_and_velocities(self):
        """
        Récupère les débits massiques et vitesses dans les conduites.

        - Si le mass flow d'entrée ne dépend pas du temps (inlet_mass_flow non callable) :
            retourne (mass_flows, velocities) où chaque élément est un vecteur 1D
            de taille (n_pipes,), indexé par l'id du pipe.

        - Si le mass flow d'entrée dépend du temps (inlet_mass_flow callable) :
            retourne (mass_flows_func, velocities_func) où chaque élément est une fonction
            f(t) -> vecteur 1D (n_pipes,) donnant les valeurs à l'instant t.
        """

        n_pipes = len(self.pipes)

        def _compute_step(t):
            mass_flows = self._compute_mass_flows(t)
            velocities = np.zeros_like(mass_flows)
            mask = (self.pipe_areas > 0.0) & (self.props.rho > 0.0)
            velocities[mask] = mass_flows[mask] / (self.props.rho * self.pipe_areas[mask])
            return mass_flows, velocities

        # Cas 1 : débit d'entrée constant → vecteurs 1D
        if not callable(self.inlet_mass_flow):
            mass_flows, velocities = _compute_step(0.0)  # n'importe quel t
            return mass_flows, velocities

        # Cas 2 : débit d'entrée variable → on renvoie des fonctions de t
        def mass_flows_func(t):
            mf, _ = _compute_step(t)
            return mf

        def velocities_func(t):
            _, vel = _compute_step(t)
            return vel

        return mass_flows_func, velocities_func
