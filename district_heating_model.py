import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from graph_utils import Graph
from config import MIN_RETURN_TEMP, PIPE_GENERATION

class Pipe:
    """
    Modélise une conduite 1D discrétisée en cellules de température.
    Gère le calcul local dT/dt en fonction du débit, des pertes et de la diffusion.
    """

    @staticmethod
    def generate_parameters(edges, dx, seed,
                            n_segments_min=None, n_segments_max=None,
                            diameter_min=None, diameter_max=None,
                            h_min=None, h_max=None):
        """
        Génère de façon reproductible les paramètres géométriques des conduites
        pour une liste d'arêtes `edges` et un `seed` donné.

        Si les paramètres ne sont pas fournis, utilise les valeurs de PIPE_GENERATION
        définies dans config.py.

        Retourne:
            lengths   : np.ndarray de longueurs (m)
            diameters : np.ndarray de diamètres (m)
            n_segments: np.ndarray de nombre de segments par conduite
            h_vals    : np.ndarray de coefficients de pertes (W/m/K) ou équivalent
        """
        # Valeurs par défaut depuis la config globale
        if n_segments_min is None:
            n_segments_min = PIPE_GENERATION["n_segments_min"]
        if n_segments_max is None:
            n_segments_max = PIPE_GENERATION["n_segments_max"]
        if diameter_min is None:
            diameter_min = PIPE_GENERATION["diameter_min"]
        if diameter_max is None:
            diameter_max = PIPE_GENERATION["diameter_max"]
        if h_min is None:
            h_min = PIPE_GENERATION["h_min"]
        if h_max is None:
            h_max = PIPE_GENERATION["h_max"]

        rng = np.random.default_rng(seed)
        n_pipes = len(edges)

        n_segments = rng.integers(n_segments_min, n_segments_max, size=n_pipes)
        lengths = dx * n_segments
        diameters = rng.uniform(diameter_min, diameter_max, size=n_pipes)
        h_vals = rng.uniform(h_min, h_max, size=n_pipes)

        return lengths, diameters, n_segments, h_vals

    def __init__(self, nodes, length, diameter, dx,
                 rho, cp, heat_loss_coeff, thermal_conductivity, external_temp):
        """
        nodes  : (u, v) identifiants des noeuds amont/aval.
        length : longueur totale de la conduite (m).
        diameter : diamètre intérieur (m).
        dx     : taille spatiale d'une cellule (m).
        rho, cp, heat_loss_coeff, thermal_conductivity, external_temp : paramètres physiques.
        """
        self.nodes = nodes
        self.length = length
        self.diameter = diameter
        self.dx = dx

        self.rho = rho
        self.cp = cp
        self.external_temp = external_temp
        self.thermal_conductivity = thermal_conductivity

        self.area = np.pi * (diameter / 2)**2
        self.loss_factor = (4.0 * heat_loss_coeff) / (diameter * rho * cp)
        self.diffusivity = (
            thermal_conductivity / (rho * cp)
            if thermal_conductivity > 0 else 0
        )
        self.n_cells = int(round(length / dx))
        if self.n_cells < 2:
            raise ValueError(f"Conduite {nodes}: Longueur insuffisante pour le pas dx choisi.")


    def compute_derivatives(self, t, T, mass_flow, inlet_temp, dT_dt):
        """
        Calcule in-place les dérivées temporelles dT_dt pour toutes les cellules.

        t          : temps courant (s).
        T          : température dans les cellules (vecteur 1D).
        mass_flow  : débit massique dans la conduite (kg/s, signe = sens).
        inlet_temp : température appliquée à la cellule d'entrée.
        dT_dt      : vecteur résultat (même shape que T) modifié sur place.
        """
        # external temperature can be a scalar or callable
        ext_temp = self.external_temp
        T_ext = ext_temp(t) if callable(ext_temp) else ext_temp
        rho = self.rho

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
            k = self.diffusivity / (self.dx**2)
            diffusion = np.zeros_like(T)
            diffusion[1:-1] = k * (T[:-2] - 2*T[1:-1] + T[2:])
            # Conditions aux bords simplifiées pour la diffusion (Neumann ou Dirichlet implicite)
            diffusion[0] = k * (T[1] - T[0])
            diffusion[-1] = k * (T[-2] - T[-1])
            dT_dt += diffusion


class DistrictHeatingNetwork:
    """
    Modélise un réseau de chaleur complet :
    - stocke toutes les conduites (Pipe) et leur discrétisation globale,
    - calcule les débits sur chaque branche à partir du graphe et des splits,
    - résout la dynamique thermique dans tout le réseau,
    - (optionnel) applique des soutirages de puissance au niveau des noeuds consommateurs.
    """

    def __init__(self, pipes: list,
                 inlet_temp=70, inlet_mass_flow=10, node_splits=None,
                 graph: Graph = None,
                 rho=1000.0, cp=4182.0,
                 node_power_funcs=None,
                 t_min_return: float = MIN_RETURN_TEMP):
        """
        pipes            : liste d'objets Pipe.
        inlet_mass_flow  : débit massique en entrée (constante ou fonction(t)).
        inlet_temp       : température en entrée (constante ou fonction(t)).
        node_splits      : dict {node_id: {child_node_id: fraction}}.
        graph            : objet Graph déjà construit (sinon reconstruit).
        rho, cp          : propriétés globales (pour vitesses, bilans énergétiques).
        node_power_funcs : dict optionnel {node_id: f(t)} donnant la puissance demandée (W).
        t_min_return     : température minimale possible après soutirage (°C).
        """

        self.pipes = pipes
        self.rho = rho
        self.cp = cp

        if graph is not None:
            self.graph = graph
        else:
            edges = [p.nodes for p in self.pipes]
            self.graph = Graph(edges)

        self.n_nodes = self.graph.get_nodes_count()
        self.topo_nodes = np.array(self.graph.get_topo_nodes(), dtype=int)

        # 2. Initialisation des slices (spécifique à la physique) et population du graphe
        self._init_pipe_slices()

        # Définition noeud d'entrée
        self.inlet_node_id = self.graph.get_inlet_node()

        # Débits / températures d'entrée (float ou callable(t))
        self.inlet_mass_flow = inlet_mass_flow
        self.inlet_temp = inlet_temp

        # Dictionnaire : node_id -> {next_node_id: fraction}
        if node_splits is None:
            node_splits = {}
        self.set_node_splits(node_splits)

        # Consommation de puissance aux noeuds (optionnelle)
        # Si node_power_funcs est None ou vide, aucune consommation n'est appliquée.
        self.node_power_funcs = node_power_funcs or {}
        self.t_min_return = t_min_return

        # Stockage de la dernière température nodale calculée (pour inspection externe)
        self.last_node_temps = None

        # Buffers pour les puissances totales (sur toute la simulation)
        # Remplis à chaque appel de compute_system_dynamics
        self.power_time = []             # temps des évaluations
        self.total_power_demand = []     # somme P_target(t) pour tous les noeuds avec demande
        self.total_power_supplied = []   # somme P_effective(t) réellement soutirée

        # Flag pour ignorer le tout premier point (état initial "froid")
        self._has_recorded_first_step = False

    def solve(self, tspan, initial_state, **kwargs):
        """
        Intègre le système d'EDO sur tspan avec solve_ivp.

        tspan              : (t0, tf).
        initial_state      : vecteur température initiale sur toutes les cellules.
        boundary_conditions: dict {node_id: valeur_ou_fonction(t)} pour les noeuds.
        kwargs             : options solve_ivp (méthode, tolérances, t_eval, ...).

        Retourne l'objet OdeResult de SciPy.
        """
        # reset des buffers de puissance pour une nouvelle simulation
        self.power_time = []
        self.total_power_demand = []
        self.total_power_supplied = []
        self._has_recorded_first_step = False

        fun = lambda t, y: self.compute_system_dynamics(t, y)
        return solve_ivp(fun, tspan, initial_state, **kwargs)    

    def compute_system_dynamics(self, t, global_state):
        """
        Fonction f(t, y) passée au solveur.

        - calcule les débits massiques dans toutes les conduites,
        - effectue le mélange aux noeuds pour obtenir la température nodale,
        - (optionnel) applique une chute de température due aux puissances soutirées,
        - appelle chaque Pipe pour obtenir dT/dt dans ses cellules.

        Retourne dy_dt (même shape que global_state).
        """

        dy_dt = np.empty_like(global_state)

        # 1. Débits massiques
        mass_flows = self._compute_mass_flows(t)

        # 2. Température aux noeuds (mélange hydraulique)
        node_temps = self._solve_nodes_temperature(global_state, mass_flows, t)

        # 3. Application optionnelle des soutirages de puissance aux noeuds
        total_p_demand = 0.0
        total_p_supplied = 0.0
        if self.node_power_funcs:
            # on a besoin des températures AVANT et APRÈS soutirage pour calculer P_effective
            parents_map = self.graph.get_parent_nodes()
            node_temps_before = node_temps.copy()

            # appliquer la chute de température
            node_temps = self._apply_node_power_consumption(t, node_temps, mass_flows)

            # calcul de la puissance totale
            for node_id, p_func in self.node_power_funcs.items():
                p_target = p_func(t)
                total_p_demand += p_target

                parents = parents_map.get(node_id, [])
                if not parents:
                    continue

                m_in = 0.0
                for parent_node in parents:
                    edge_data = self.graph.edges[(parent_node, node_id)]
                    pipe_idx = edge_data["pipe_index"]
                    m_in += mass_flows[pipe_idx]

                if m_in <= 0.0:
                    continue

                node_idx = self.graph.get_id_from_node(node_id)
                T_in = node_temps_before[node_idx]
                T_out = node_temps[node_idx]

                # puissance réellement soutirée à ce noeud (bornée implicitement par t_min_return)
                p_effective = self.cp * m_in * max(T_in - T_out, 0.0)
                total_p_supplied += p_effective

        # stock pour inspection depuis l'extérieur
        self.last_node_temps = node_temps

        # remplir les buffers de puissance pour ce pas de temps
        # on ignore le tout premier appel, où l'état initial est entièrement à MIN_RETURN_TEMP
        if not self._has_recorded_first_step:
            # on active le flag, mais on ne stocke pas ce point (p_supplied_tot = 0 physique)
            self._has_recorded_first_step = True
        else:
            self.power_time.append(float(t))
            self.total_power_demand.append(total_p_demand)
            self.total_power_supplied.append(total_p_supplied)

        # 4. Physique par conduite
        for i, p in enumerate(self.pipes):
            u_idx = self._u_idxs[i]
            v_idx = self._v_idxs[i]
            m_dot = mass_flows[i]

            temp_inlet = node_temps[u_idx] if m_dot >= 0 else node_temps[v_idx]
            sl = self.pipe_slices[i]
            p.compute_derivatives(t, global_state[sl], m_dot, temp_inlet, dy_dt[sl])
        return dy_dt

    def _init_pipe_slices(self):
        """
        Construit le mapping entre cellules globales et conduites :

        - pipe_slices[i]      : slice des cellules globales appartenant au pipe i,
        - idx_pipe_start/end  : indices globaux des premières / dernières cellules,
        - _u_idxs / _v_idxs   : index des noeuds amont/aval (pour le mélange),
        - met à jour graph.edges[(u,v)] avec toutes les infos (pipe, cells, indices).
        """
        self.pipe_areas = np.zeros(len(self.pipes)) 
        self.pipe_slices = []   # liste de slices
        self.idx_pipe_start = [] 
        self.idx_pipe_end = []
        
        # Tableaux internes pour la vectorisation (remplace les attributs publics supprimés)
        self._u_idxs = np.zeros(len(self.pipes), dtype=int)
        self._v_idxs = np.zeros(len(self.pipes), dtype=int)
        
        offset = 0

        for i, pipe in enumerate(self.pipes):
            self.pipe_areas[i] = pipe.area
            sl = slice(offset, offset + pipe.n_cells)
            self.pipe_slices.append(sl)
            self.idx_pipe_start.append(offset)
            self.idx_pipe_end.append(offset + pipe.n_cells - 1)
            
            u, v = pipe.nodes
            u_idx = self.graph.get_id_from_node(u)
            v_idx = self.graph.get_id_from_node(v)
            
            self._u_idxs[i] = u_idx
            self._v_idxs[i] = v_idx

            # Intégration des infos du pipe dans le graphe
            self.graph.edges[pipe.nodes].update({
                    "pipe_obj": pipe,
                    "pipe_index": i,
                    "length": pipe.length,
                    "diameter": pipe.diameter,
                    "cell_ids": list(range(offset, offset + pipe.n_cells)),
                    "slice": sl,
                    "u_idx": u_idx,
                    "v_idx": v_idx
                })

            offset += pipe.n_cells

        self.total_cells = offset
        self.idx_pipe_start = np.array(self.idx_pipe_start)
        self.idx_pipe_end = np.array(self.idx_pipe_end)

    def set_node_splits(self, node_splits):
        """
        Met à jour dans graph.edges la fraction de débit attribuée à chaque arc sortant.

        node_splits : dict {u: {v: fraction}} au niveau logique des noeuds.
        Si un noeud n'est pas spécifié, on répartit uniformément entre ses enfants.
        """
        # Reset
        for u, v in self.graph.edges:
            self.graph.edges[(u,v)]["split_fraction"] = 0.0

        child_nodes = self.graph.get_child_nodes()

        for u_logical, children in child_nodes.items():
            if not children:
                continue

            splits = node_splits.get(u_logical, None)

            if splits is None:
                # Uniforme
                val = 1.0 / len(children)
                for v_logical in children:
                    self.graph.edges[(u_logical, v_logical)]["split_fraction"] = val
            else:
                # Spécifique
                total = 0.0
                for v_logical in children:
                    frac = splits.get(v_logical, 0.0)
                    self.graph.edges[(u_logical, v_logical)]["split_fraction"] = frac
                    total += frac
                
                # Normalisation
                if total > 0:
                    for v_logical in children:
                        self.graph.edges[(u_logical, v_logical)]["split_fraction"] /= total
                else:
                    val = 1.0 / len(children)
                    for v_logical in children:
                        self.graph.edges[(u_logical, v_logical)]["split_fraction"] = val

    def _compute_mass_flows(self, t):
        """
        Propage le débit massique depuis le noeud d'entrée sur tout le graphe.

        - lit les fractions 'split_fraction' dans graph.edges,
        - remplit un vecteur mass_flows[i] pour chaque pipe,
        - stocke aussi 'mass_flow' dans graph.edges[(u,v)].

        Retourne mass_flows (np.ndarray de taille n_pipes).
        """
        n_pipes = len(self.pipes)
        mass_flows = np.zeros(n_pipes, dtype=float)
        node_mass_in = np.zeros(self.n_nodes, dtype=float)

        # Débit en entrée du réseau
        inlet_idx = self.graph.get_id_from_node(self.inlet_node_id)
        inlet_val = float(self.inlet_mass_flow(t)) if callable(self.inlet_mass_flow) else float(self.inlet_mass_flow)
        node_mass_in[inlet_idx] += inlet_val

        child_nodes = self.graph.get_child_nodes()

        # Propagation selon l'ordre topologique
        for node_idx in self.topo_nodes:
            incoming = node_mass_in[node_idx]
            if incoming <= 0.0:
                continue

            u_logical = self.graph.get_node_from_id(node_idx)
            for v_logical in child_nodes.get(u_logical, []):
                edge_key = (u_logical, v_logical)
                edge_data = self.graph.edges[edge_key]

                pipe_idx = edge_data["pipe_index"]
                frac = edge_data.get("split_fraction", 0.0)

                m_dot = incoming * frac
                mass_flows[pipe_idx] = m_dot
                edge_data["mass_flow"] = m_dot  # mémorisation dans le graphe

                v_idx = edge_data["v_idx"]
                node_mass_in[v_idx] += m_dot

        return mass_flows

    def _solve_nodes_temperature(self, global_state, mass_flows, t):
        """
        Calcule la température de mélange à chaque noeud à partir :
        - des débits massiques dans chaque conduite,
        - de la température en entrée/sortie de conduite selon le sens du flux,
        Puis impose la température au noeud d'entrée
        """
        n_nodes = self.n_nodes

        # Accumulateurs de masse et d'enthalpie entrantes sur chaque noeud
        node_mass = np.zeros(n_nodes)
        node_heat = np.zeros(n_nodes)

        # Masques de sens du flux
        flow_pos = mass_flows > 0.0    # u -> v
        flow_neg = ~flow_pos           # v -> u

        # --- Flux positifs (u -> v) : contribution à v avec T en fin de conduite ---
        if np.any(flow_pos):
            target_nodes = self._v_idxs[flow_pos]
            m_vals = mass_flows[flow_pos]
            T_out = global_state[self.idx_pipe_end[flow_pos]]
            np.add.at(node_mass, target_nodes, m_vals)
            np.add.at(node_heat, target_nodes, m_vals * T_out)

        # --- Flux négatifs (v -> u) : contribution à u avec T en début de conduite ---
        if np.any(flow_neg):
            target_nodes = self._u_idxs[flow_neg]
            m_vals = -mass_flows[flow_neg]
            T_out = global_state[self.idx_pipe_start[flow_neg]]
            np.add.at(node_mass, target_nodes, m_vals)
            np.add.at(node_heat, target_nodes, m_vals * T_out)

        # Température moyenne pondérée par la masse
        with np.errstate(divide="ignore", invalid="ignore"):
            node_temps = node_heat / node_mass

        node_temps[np.isnan(node_temps)] = 10.0

        # Imposer la température au noeud d'entrée (producteur)
        inlet_idx = self.graph.get_id_from_node(self.inlet_node_id)
        T_in = self.inlet_temp(t) if callable(self.inlet_temp) else float(self.inlet_temp)
        node_temps[inlet_idx] = T_in

        return node_temps

    def _apply_node_power_consumption(self, t, node_temps, mass_flows):
        """
        Applique, pour chaque noeud ayant une demande de puissance, une chute de température
        locale: T_out = T_in - P / (m_in * cp), bornée par t_min_return.

        - node_temps : températures au noeud avant soutirage (après mélange),
        - mass_flows : débits massiques dans chaque conduite (kg/s).
        Retourne un nouveau vecteur node_temps modifié.
        """
        node_temps = node_temps.copy()
        cp = self.cp
        t_min_return = self.t_min_return

        parents_map = self.graph.get_parent_nodes()

        for node_id, p_func in self.node_power_funcs.items():
            # débit entrant au noeud = somme des débits de tous les tuyaux qui y arrivent
            parents = parents_map.get(node_id, [])
            if not parents:
                continue

            m_in = 0.0
            for parent_node in parents:
                edge_data = self.graph.edges[(parent_node, node_id)]
                pipe_idx = edge_data["pipe_index"]
                m_in += mass_flows[pipe_idx]

            if m_in <= 0.0:
                continue

            node_idx = self.graph.get_id_from_node(node_id)
            T_node_in = node_temps[node_idx]

            # demande de puissance à cet instant
            p_target = p_func(t)

            # puissance maximale possible sans descendre sous t_min_return
            delta_T_max = max(T_node_in - t_min_return, 0.0)
            p_max = m_in * cp * delta_T_max

            p_supplied = min(p_target, p_max)

            delta_T = p_supplied / (m_in * cp) if m_in * cp > 0 else 0.0
            T_node_out = max(T_node_in - delta_T, t_min_return)

            # on met simplement à jour la température nodale
            node_temps[node_idx] = T_node_out

        return node_temps

    def get_pipes_temperature(self, solution_y):
        """
        Renvoie la moyenne de la température dans chaque conduite à chaque instant.

        solution_y : array (total_cells, n_times) retourné par solve_ivp.y.

        Retourne un array (n_pipes, n_times) avec la température moyenne de chaque pipe.
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
        Fournit les débits massiques et vitesses dans chaque conduite.

        - Si inlet_mass_flow est constant : retourne (mass_flows, velocities),
        - Si inlet_mass_flow est fonction du temps :
          retourne (mass_flows_func, velocities_func) qui évaluent ces grandeurs à t.
        """
        def _compute_step(t):
            mass_flows = self._compute_mass_flows(t)
            velocities = mass_flows / (self.rho * self.pipe_areas)
            return mass_flows, velocities

        if callable(self.inlet_mass_flow):
            # Débit d'entrée variable → on renvoie des fonctions
            def mass_flows_func(t):
                mf, _ = _compute_step(t)
                return mf

            def velocities_func(t):
                _, v = _compute_step(t)
                return v

            return mass_flows_func, velocities_func

        # Débit d'entrée constant → un seul calcul suffit
        return _compute_step(0.0)

    def get_total_power_demand(self, t_eval):
        """
        Retourne la demande totale de puissance (somme sur tous les noeuds) interpolée sur t_eval.

        Interpolation en escalier ("previous") pour conserver la nature step des profils.
        """
        if not self.power_time:
            return np.zeros_like(t_eval, dtype=float)

        t_arr = np.array(self.power_time)
        p_arr = np.array(self.total_power_demand)

        f = interp1d(
            t_arr,
            p_arr,
            kind="previous",
            bounds_error=False,
            fill_value=(p_arr[0], p_arr[-1]),
        )
        return f(t_eval)

    def get_total_power_supplied(self, t_eval):
        """
        Retourne la puissance totale effectivement soutirée (somme sur tous les noeuds),
        interpolée sur t_eval.

        Interpolation en escalier ("previous") pour conserver une forme en marche.
        """
        if not self.power_time:
            return np.zeros_like(t_eval, dtype=float)

        t_arr = np.array(self.power_time)
        p_arr = np.array(self.total_power_supplied)

        f = interp1d(
            t_arr,
            p_arr,
            kind="previous",
            bounds_error=False,
            fill_value=(p_arr[0], p_arr[-1]),
        )
        return f(t_eval)

    def _evaluate_inlet_signals(self, t_eval):
        """
        Évalue les signaux d'entrée (débit massique et température source) sur t_eval.

        t_eval : float ou array-like de temps (s).

        Retourne:
            m_dot_t : np.ndarray de même shape que t_eval (kg/s)
            T_in_t  : np.ndarray de même shape que t_eval (°C)
        """
        t_arr = np.atleast_1d(t_eval)

        # Débit
        if callable(self.inlet_mass_flow):
            m_dot_t = np.array([float(self.inlet_mass_flow(ti)) for ti in t_arr], dtype=float)
        else:
            m_dot_t = np.full_like(t_arr, float(self.inlet_mass_flow), dtype=float)

        # Température
        if callable(self.inlet_temp):
            T_in_t = np.array([float(self.inlet_temp(ti)) for ti in t_arr], dtype=float)
        else:
            T_in_t = np.full_like(t_arr, float(self.inlet_temp), dtype=float)

        return m_dot_t, T_in_t

    def get_boiler_power(self, t_eval):
        """
        Retourne la puissance fournie par la chaudière P_boiler(t) sur t_eval.

        Convention identique à l'env Gym:
            P_boiler(t) = m_dot(t) * cp * (T_in(t) - MIN_RETURN_TEMP)

        t_eval : float ou array-like (s).
        Retourne un np.ndarray (même shape que t_eval) en Watts.
        """
        t_arr = np.atleast_1d(t_eval)
        m_dot_t, T_in_t = self._evaluate_inlet_signals(t_arr)
        p_boiler = m_dot_t * self.cp * (T_in_t - MIN_RETURN_TEMP)
        return p_boiler if np.ndim(t_eval) else float(p_boiler[0])

    def get_pump_power(self, t_eval):
        """
        Retourne la puissance de pompage P_pump(t) sur t_eval.

        Même convention que dans l'environnement Gym:
            P_pump(t) = 1000.0 * m_dot(t)
        """
        t_arr = np.atleast_1d(t_eval)
        m_dot_t, _ = self._evaluate_inlet_signals(t_arr)
        p_pump = 1000.0 * m_dot_t
        return p_pump if np.ndim(t_eval) else float(p_pump[0])
