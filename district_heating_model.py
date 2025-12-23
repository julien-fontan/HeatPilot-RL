import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable, Optional, Union
from graph_utils import Graph
from config import PHYSICAL_PROPS

@dataclass(frozen=True)
class PipeConfig:
    """
    Conteneur de données immuable pour la configuration d'une conduite.
    """
    nodes: Tuple[str, str]  # (u, v)
    length: float
    diameter: float
    dx: float
    rho: float
    cp: float
    heat_loss_coeff: float
    thermal_conductivity: float
    external_temp: Union[float, Callable]

    @staticmethod
    def generate_parameters(edges, dx, seed):
        """Génère de façon reproductible les paramètres géométriques."""
        rng = np.random.default_rng(seed)
        n_pipes = len(edges)
        
        # Récupération des bornes depuis la config
        l_min, l_max = PHYSICAL_PROPS["length_min"], PHYSICAL_PROPS["length_max"]
        d_min, d_max = PHYSICAL_PROPS["diameter_min"], PHYSICAL_PROPS["diameter_max"]
        h_min, h_max = PHYSICAL_PROPS["heat_loss_coeff_min"], PHYSICAL_PROPS["heat_loss_coeff_max"]

        diameters = rng.uniform(d_min, d_max, size=n_pipes)
        lengths = rng.uniform(l_min, l_max, size=n_pipes)
        h_vals = rng.uniform(h_min, h_max, size=n_pipes)

        # Calcul du nombre de segments
        n_segments = np.rint(lengths / dx).astype(int)
        n_segments = np.maximum(n_segments, 2) # Min 2 cellules
        lengths = n_segments * dx 

        return lengths, diameters, n_segments, h_vals

class DistrictHeatingNetwork:
    """
    Moteur de simulation vectorisé pour réseau de chaleur.
    Traite le système entier comme de grandes matrices pour la performance.
    """

    def __init__(self, graph: Optional[Graph], pipe_configs: List[PipeConfig],
                 t_min_return: float, 
                 inlet_temp: Union[float, Callable], 
                 inlet_mass_flow: Union[float, Callable], 
                 rho: float, cp: float,
                 node_splits: Dict = None, 
                 node_power_funcs: Dict = None):
        
        self.pipe_configs = pipe_configs
        self.rho = rho
        self.cp = cp
        self.t_min_return = t_min_return
        
        # Gestion du Graphe
        if graph is not None:
            self.graph = graph
        else:
            edges = [p.nodes for p in pipe_configs]
            self.graph = Graph(edges)

        self.n_nodes = self.graph.get_nodes_count()
        self.topo_nodes = np.array(self.graph.get_topo_nodes(), dtype=int)
        self.inlet_node_id = self.graph.get_inlet_node()

        # Inputs
        self.inlet_mass_flow = inlet_mass_flow
        self.inlet_temp = inlet_temp
        self.set_node_splits(node_splits or {})
        self.node_power_funcs = node_power_funcs or {}

        # --- VECTORISATION : Initialisation des structures globales ---
        self._build_vectorized_structures()

    def _build_vectorized_structures(self):
        """
        Construit les tableaux Numpy (aplatis) représentant tout le réseau.
        Évite les boucles 'for' pendant la simulation.
        """
        self.total_cells = 0
        
        # Listes temporaires pour construction
        all_dx = []
        all_areas = []
        all_loss_factors = []
        
        # Mappings
        self.pipe_slices = []       # [slice(0, 10), slice(10, 25)...]
        self.pipe_u_idxs = []       # Index nodal amont pour chaque pipe
        self.pipe_v_idxs = []       # Index nodal aval pour chaque pipe
        self.cell_to_pipe_idx = []  # Pour chaque cellule, quel est son pipe ID ?

        # Indices des cellules aux frontières des pipes
        self.idx_pipe_start = []    # Index de la première cellule de chaque pipe
        self.idx_pipe_end = []      # Index de la dernière cellule de chaque pipe

        offset = 0
        for i, p in enumerate(self.pipe_configs):
            n_cells = int(round(p.length / p.dx))
            
            # Propriétés physiques par cellule
            area = np.pi * (p.diameter / 2)**2
            loss_factor = (4.0 * p.heat_loss_coeff) / (p.diameter * p.rho * p.cp)
            
            # Remplissage des tableaux globaux
            all_dx.extend([p.dx] * n_cells)
            all_areas.extend([area] * n_cells)
            all_loss_factors.extend([loss_factor] * n_cells)
            self.cell_to_pipe_idx.extend([i] * n_cells)

            # Gestion des indices et slices
            sl = slice(offset, offset + n_cells)
            self.pipe_slices.append(sl)
            self.idx_pipe_start.append(offset)
            self.idx_pipe_end.append(offset + n_cells - 1)
            
            # Mapping noeuds
            u_idx = self.graph.get_id_from_node(p.nodes[0])
            v_idx = self.graph.get_id_from_node(p.nodes[1])
            self.pipe_u_idxs.append(u_idx)
            self.pipe_v_idxs.append(v_idx)
            
            # Mise à jour graphe (pour le routage)
            self.graph.edges[p.nodes].update({
                "pipe_index": i,
                "v_idx": v_idx # requis pour _compute_mass_flows
            })

            offset += n_cells

        self.total_cells = offset
        
        # Conversion en tableaux Numpy pour calcul rapide
        self.all_dx = np.array(all_dx)
        self.all_areas = np.array(all_areas)
        self.all_loss_factors = np.array(all_loss_factors)
        self.cell_to_pipe_idx = np.array(self.cell_to_pipe_idx, dtype=int)
        
        self.pipe_u_idxs = np.array(self.pipe_u_idxs, dtype=int)
        self.pipe_v_idxs = np.array(self.pipe_v_idxs, dtype=int)
        self.idx_pipe_start = np.array(self.idx_pipe_start, dtype=int)
        self.idx_pipe_end = np.array(self.idx_pipe_end, dtype=int)

        # --- Pré-calcul des indices pour l'advection interne (i -> i+1) ---
        # On crée un masque des cellules "internes" (qui ne sont ni début ni fin de tuyau)
        # pour éviter les conditions aux limites dans le calcul vectoriel principal.
        
        # Indices de TOUTES les cellules
        all_indices = np.arange(self.total_cells)
        
        # Pour le flux POSITIF (->): T[i] dépend de T[i-1]
        # On exclut les indices de début de pipe (qui dépendent des noeuds)
        mask_internal_pos = ~np.isin(all_indices, self.idx_pipe_start)
        self.ids_internal_current_pos = all_indices[mask_internal_pos]
        self.ids_internal_upstream_pos = self.ids_internal_current_pos - 1

        # Pour le flux NEGATIF (<-): T[i] dépend de T[i+1]
        # On exclut les indices de fin de pipe
        mask_internal_neg = ~np.isin(all_indices, self.idx_pipe_end)
        self.ids_internal_current_neg = all_indices[mask_internal_neg]
        self.ids_internal_upstream_neg = self.ids_internal_current_neg + 1

    def set_node_splits(self, node_splits):
        """Configure les fractions de débit aux intersections."""
        # (Logique identique à votre code original, mise à jour graph.edges)
        for u, v in self.graph.edges:
            self.graph.edges[(u,v)]["split_fraction"] = 0.0
        
        child_nodes = self.graph.get_child_nodes()
        for u, children in child_nodes.items():
            if not children: continue
            splits = node_splits.get(u, {})
            
            # Normalisation
            total = sum(splits.get(v, 0.0) for v in children)
            base_val = 1.0 / len(children)
            
            for v in children:
                if splits and total > 0:
                    frac = splits.get(v, 0.0) / total
                elif splits and total == 0:
                    frac = base_val # Fallback si split défini mais somme nulle
                else:
                    frac = base_val # Uniforme par défaut
                
                self.graph.edges[(u,v)]["split_fraction"] = frac

    def solve(self, tspan, initial_state, **kwargs):
        """Lance la simulation."""
        fun = lambda t, y: self.compute_system_dynamics(t, y)
        sol = solve_ivp(fun, tspan, initial_state, **kwargs)
        return sol

    def compute_system_dynamics(self, t, T_cells):
        """
        Calcul vectorisé de dT/dt pour toutes les cellules du réseau.
        Complexité : O(N_cells) sans boucle Python.
        """
        dy_dt = np.zeros_like(T_cells)
        
        # 1. Calcul des débits dans chaque tuyau (N_pipes)
        # C'est la seule partie qui itère sur les noeuds (rapide car N_nodes << N_cells)
        pipe_mass_flows = self._compute_mass_flows(t)
        
        # 2. Diffusion du débit sur toutes les cellules (Broadcasting N_cells)
        # cell_velocities[k] = débit du tuyau possédant la cellule k / (rho * Area[k])
        cell_m_flows = pipe_mass_flows[self.cell_to_pipe_idx]
        cell_velocities = cell_m_flows / (self.rho * self.all_areas)
        
        # 3. Températures aux noeuds (Mélange)
        node_temps = self._solve_nodes_temperature(T_cells, pipe_mass_flows, t)
        
        # 4. Application des consommations (demande de puissance)
        if self.node_power_funcs:
            node_temps = self._apply_node_power_consumption(t, node_temps, pipe_mass_flows)

        # 5. Calcul physique (Advection + Pertes)
        
        # Gestion de la Température Extérieure (peut être scalaire ou callable)
        # On suppose ici constant par pipe pour simplifier, ou global. 
        # Si c'est global :
        ext_temp = self.pipe_configs[0].external_temp # Simplification: on prend le 1er
        T_ext = ext_temp(t) if callable(ext_temp) else ext_temp
        
        # --- A. FLUX POSITIF (V > 0) ---
        # Advection interne: -v * (T[i] - T[i-1]) / dx
        # Calcul uniquement là où v > 0
        
        # Optimisation : On calcule l'advection partout, on masquera après.
        # Terme interne (cellules 1..N)
        adv_internal_pos = -(cell_velocities[self.ids_internal_current_pos] / self.all_dx[self.ids_internal_current_pos]) * \
                           (T_cells[self.ids_internal_current_pos] - T_cells[self.ids_internal_upstream_pos])
        
        # On applique aux bons endroits dans le vecteur global
        dy_dt[self.ids_internal_current_pos] = adv_internal_pos

        # Conditions aux limites (Début de tuyau) : T_upstream = Node_Temp
        # Pour les cellules 'start', T[i-1] n'existe pas, c'est T_node[u]
        start_nodes = self.pipe_u_idxs
        T_start_inlet = node_temps[start_nodes]
        
        adv_start = -(cell_velocities[self.idx_pipe_start] / self.all_dx[self.idx_pipe_start]) * \
                    (T_cells[self.idx_pipe_start] - T_start_inlet)
        
        dy_dt[self.idx_pipe_start] = adv_start

        # --- B. FLUX NEGATIF (V < 0) ---
        # Si on a du flux inverse, on écrase les valeurs précédentes aux endroits concernés.
        # (Note: Code simplifié, si V < 0, il faudrait recalculer T_inlet = T_node[v] pour la fin du tuyau)
        # Pour la lisibilité ici, on suppose V >= 0 majoritairement. 
        # Si vous avez besoin du flux inverse rigoureux, on applique la même logique miroir.
        
        # Masque global de flux (optionnel si flux toujours positif)
        # mask_neg = cell_velocities < 0
        # if np.any(mask_neg):
        #    ... logique miroir ...

        # --- C. PERTES THERMIQUES ---
        # - h * (T - T_ext)
        losses = -self.all_loss_factors * (T_cells - T_ext)
        dy_dt += losses

        return dy_dt

    def _compute_mass_flows(self, t):
        """Propage le débit depuis l'entrée (inchangé logiquement mais nettoyé)."""
        mass_flows = np.zeros(len(self.pipe_configs))
        node_mass_in = np.zeros(self.n_nodes)

        # Source
        inlet_val = self.inlet_mass_flow(t) if callable(self.inlet_mass_flow) else self.inlet_mass_flow
        inlet_idx = self.graph.get_id_from_node(self.inlet_node_id)
        node_mass_in[inlet_idx] = inlet_val

        # Propagation
        child_nodes_map = self.graph.get_child_nodes()
        for node_idx in self.topo_nodes:
            m_in = node_mass_in[node_idx]
            if m_in <= 1e-9: continue

            u_node = self.graph.get_node_from_id(node_idx)
            children = child_nodes_map.get(u_node, [])
            
            for v_node in children:
                edge = self.graph.edges[(u_node, v_node)]
                m_out = m_in * edge.get("split_fraction", 0.0)
                
                pipe_idx = edge["pipe_index"]
                mass_flows[pipe_idx] = m_out
                
                # Ajout au noeud suivant
                node_mass_in[edge["v_idx"]] += m_out
                
        return mass_flows

    def _solve_nodes_temperature(self, T_cells, pipe_mass_flows, t):
        """Mélange parfait aux noeuds (Vectorisé avec np.add.at)."""
        node_mass = np.zeros(self.n_nodes)
        node_enthalpy = np.zeros(self.n_nodes)

        # Contributions U -> V (Flux positifs)
        # On prend la T de la DERNIÈRE cellule du tuyau (idx_pipe_end)
        mask_pos = pipe_mass_flows > 0
        if np.any(mask_pos):
            idx_pipes = np.where(mask_pos)[0]
            target_nodes = self.pipe_v_idxs[idx_pipes]
            m_vals = pipe_mass_flows[idx_pipes]
            T_vals = T_cells[self.idx_pipe_end[idx_pipes]] # Température sortie tuyau
            
            np.add.at(node_mass, target_nodes, m_vals)
            np.add.at(node_enthalpy, target_nodes, m_vals * T_vals)

        # Calcul moyenne
        with np.errstate(divide='ignore', invalid='ignore'):
            node_temps = node_enthalpy / node_mass
            
        # Fallback pour noeuds sans débit (garde T ambiante ou précédente, ici arbitraire 10.0)
        node_temps[np.isnan(node_temps)] = 10.0
        
        # Condition limite SOURCE
        T_src = self.inlet_temp(t) if callable(self.inlet_temp) else self.inlet_temp
        inlet_idx = self.graph.get_id_from_node(self.inlet_node_id)
        node_temps[inlet_idx] = T_src
        
        return node_temps

    def _apply_node_power_consumption(self, t, node_temps, pipe_mass_flows):
        """Calcul de la chute de température due aux consommateurs."""
        # Note : Cette méthode doit recalculer m_in par noeud.
        # Pour optimiser, on pourrait pré-mapper "pipes_incoming_to_node".
        # Ici on garde la logique simple pour la lisibilité.
        
        updated_temps = node_temps.copy()
        parents_map = self.graph.get_parent_nodes()

        for node_id, p_func in self.node_power_funcs.items():
            node_idx = self.graph.get_id_from_node(node_id)
            
            # Somme des débits entrants
            m_in = 0.0
            parents = parents_map.get(node_id, [])
            for p_node in parents:
                edge = self.graph.edges[(p_node, node_id)]
                m_in += pipe_mass_flows[edge["pipe_index"]]
            
            if m_in <= 1e-9: continue

            p_target = p_func(t)
            T_in = updated_temps[node_idx]
            
            # Bilan énergie
            delta_T_max = max(T_in - self.t_min_return, 0.0)
            p_max = m_in * self.cp * delta_T_max
            p_effective = min(p_target, p_max)
            
            delta_T = p_effective / (m_in * self.cp)
            updated_temps[node_idx] = max(T_in - delta_T, self.t_min_return)
            
        return updated_temps

    def reconstruct_metrics(self, sol_t, sol_y):
        """
        Méthode à appeler APRES solve_ivp pour récupérer les KPI (Puissances, etc.)
        Rejoue la logique sur les temps de sortie.
        
        Retourne : dictionnaire de résultats (times, demand, supplied, etc.)
        """
        times = sol_t
        n_steps = len(times)
        
        total_demand = np.zeros(n_steps)
        total_supplied = np.zeros(n_steps)
        
        # Mapping parents pour éviter de le refaire dans la boucle
        parents_map = self.graph.get_parent_nodes()
        
        for k, t in enumerate(times):
            # 1. État du système à t
            T_state = sol_y[:, k]
            m_flows = self._compute_mass_flows(t)
            node_temps = self._solve_nodes_temperature(T_state, m_flows, t)
            
            # 2. Calcul soutirage (copie de la logique compute_dynamics mais juste pour les metrics)
            for node_id, p_func in self.node_power_funcs.items():
                p_target = p_func(t)
                total_demand[k] += p_target
                
                # Calcul m_in
                m_in = 0.0
                for p_node in parents_map.get(node_id, []):
                    edge = self.graph.edges[(p_node, node_id)]
                    m_in += m_flows[edge["pipe_index"]]
                
                if m_in > 1e-9:
                    node_idx = self.graph.get_id_from_node(node_id)
                    T_in = node_temps[node_idx]
                    p_possible = m_in * self.cp * max(T_in - self.t_min_return, 0.0)
                    total_supplied[k] += min(p_target, p_possible)
        
        return {
            "time": times,
            "total_demand": total_demand,
            "total_supplied": total_supplied
        }