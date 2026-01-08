import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable, Optional, Union
from graph_utils import Graph
from config import PHYSICAL_PROPS

@dataclass(frozen=True)
class PipeConfig:
    nodes: Tuple[str, str]
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
        rng = np.random.default_rng(seed)
        n_pipes = len(edges)
        l_min, l_max = PHYSICAL_PROPS["length_min"], PHYSICAL_PROPS["length_max"]
        d_min, d_max = PHYSICAL_PROPS["diameter_min"], PHYSICAL_PROPS["diameter_max"]
        h_min, h_max = PHYSICAL_PROPS["heat_loss_coeff_min"], PHYSICAL_PROPS["heat_loss_coeff_max"]

        diameters = rng.uniform(d_min, d_max, size=n_pipes)
        lengths = rng.uniform(l_min, l_max, size=n_pipes)
        h_vals = rng.uniform(h_min, h_max, size=n_pipes)

        n_segments = np.rint(lengths / dx).astype(int)
        n_segments = np.maximum(n_segments, 2)
        lengths = n_segments * dx 
        return lengths, diameters, n_segments, h_vals

def create_pipe_configs(edges, lengths, diameters, h_vals, dx, props):
    pipe_configs = []
    edge_list = edges if isinstance(edges, list) else list(edges.keys())
    for i, (u, v) in enumerate(edge_list):
        pc = PipeConfig(
            nodes=(u, v),
            length=float(lengths[i]),
            diameter=float(diameters[i]),
            dx=dx,
            rho=props["rho"],
            cp=props["cp"],
            heat_loss_coeff=float(h_vals[i]),
            thermal_conductivity=props["thermal_conductivity"],
            external_temp=props["external_temp"]
        )
        pipe_configs.append(pc)
    return pipe_configs

class DistrictHeatingNetwork:
    """
    Moteur de simulation vectorisé optimisé.
    """
    def __init__(self, graph: Graph, pipe_configs: List[PipeConfig],
                 t_min_return: float, 
                 inlet_temp: Union[float, Callable], 
                 inlet_mass_flow: Union[float, Callable], 
                 rho: float, cp: float,
                 node_splits: Dict = None, 
                 node_power_funcs: Dict = None):
        
        self.graph = graph
        self.pipe_configs = pipe_configs
        self.rho = rho
        self.cp = cp
        self.t_min_return = t_min_return
        
        self.inlet_temp = inlet_temp
        self.inlet_mass_flow = inlet_mass_flow
        self.node_power_funcs = node_power_funcs or {}

        # Vectorisation
        self._build_vectorized_structures()
        
        # Matrice de splits dense
        self.split_matrix = np.zeros((self.graph.n_nodes, self.graph.n_nodes), dtype=np.float32)
        self.set_node_splits(node_splits or {})

        # Mapping Consommateurs
        self.consumer_indices = []
        self.consumer_ids_str = []
        if self.node_power_funcs:
            for node_str in sorted(self.node_power_funcs.keys()):
                self.consumer_ids_str.append(node_str)
                self.consumer_indices.append(self.graph.get_id(node_str))
        self.consumer_indices = np.array(self.consumer_indices, dtype=int)
        
        # Mapping Terminaux
        self.terminal_indices = np.array([self.graph.get_id(n) for n in self.graph.terminal_nodes], dtype=int)

    def update_power_profiles(self, new_power_funcs: Dict):
        self.node_power_funcs = new_power_funcs

    def reset_state(self, initial_temp: float) -> np.ndarray:
        return np.full(self.total_cells, initial_temp)

    def _build_vectorized_structures(self):
        self.total_cells = 0
        all_dx, all_areas, all_loss_factors = [], [], []
        
        self.cell_to_pipe_idx = []  
        self.idx_pipe_start = []    
        self.idx_pipe_end = []      
        self.pipe_u_idxs = []       
        self.pipe_v_idxs = []       

        offset = 0
        for i, p in enumerate(self.pipe_configs):
            self.graph.register_pipe_index(p.nodes[0], p.nodes[1], i)
            n_cells = int(round(p.length / p.dx))
            area = np.pi * (p.diameter / 2)**2
            loss_factor = (4.0 * p.heat_loss_coeff) / (p.diameter * p.rho * p.cp)
            
            all_dx.extend([p.dx] * n_cells)
            all_areas.extend([area] * n_cells)
            all_loss_factors.extend([loss_factor] * n_cells)
            self.cell_to_pipe_idx.extend([i] * n_cells)

            self.idx_pipe_start.append(offset)
            self.idx_pipe_end.append(offset + n_cells - 1)
            
            self.pipe_u_idxs.append(self.graph.get_id(p.nodes[0]))
            self.pipe_v_idxs.append(self.graph.get_id(p.nodes[1]))

            offset += n_cells

        self.total_cells = offset
        self.all_dx = np.array(all_dx)
        self.all_areas = np.array(all_areas)
        self.all_loss_factors = np.array(all_loss_factors)
        self.cell_to_pipe_idx = np.array(self.cell_to_pipe_idx, dtype=int)
        self.pipe_u_idxs = np.array(self.pipe_u_idxs, dtype=int)
        self.pipe_v_idxs = np.array(self.pipe_v_idxs, dtype=int)
        self.idx_pipe_start = np.array(self.idx_pipe_start, dtype=int)
        self.idx_pipe_end = np.array(self.idx_pipe_end, dtype=int)

        all_indices = np.arange(self.total_cells)
        mask_internal = ~np.isin(all_indices, self.idx_pipe_start)
        self.ids_internal_curr = all_indices[mask_internal]
        self.ids_internal_up = self.ids_internal_curr - 1
        
        self.ext_temp_val_or_func = self.pipe_configs[0].external_temp

    def set_node_splits(self, node_splits):
        self.split_matrix.fill(0.0)
        for u_idx, children_data in enumerate(self.graph.int_adjacency):
            if not children_data: continue
            u_node = self.graph.get_node(u_idx)
            splits = node_splits.get(u_node, {})
            
            total_spec = sum(splits.get(self.graph.get_node(c_idx), 0.0) for c_idx, _ in children_data)
            count_children = len(children_data)
            base_val = 1.0 / count_children
            
            for child_idx, _ in children_data:
                child_node = self.graph.get_node(child_idx)
                if splits and total_spec > 0:
                    val = splits.get(child_node, 0.0) / total_spec
                elif splits and total_spec == 0:
                     val = base_val
                else:
                    val = base_val
                self.split_matrix[u_idx, child_idx] = val

    def compute_heuristic_splits(self, t):
        heuristic_splits = {}
        for b_idx in self.graph.branching_indices:
            children_data = self.graph.int_adjacency[b_idx] 
            if len(children_data) < 2: continue
                
            b_node_str = self.graph.get_node(b_idx)
            branch_demands = []
            child_nodes = []
            
            for child_idx, _ in children_data:
                consumers = self.graph.get_downstream_consumers_indices(b_idx, child_idx)
                d_val = 0.0
                for c_idx in consumers:
                    c_name = self.graph.get_node(c_idx)
                    d_val += self.node_power_funcs[c_name](t)
                branch_demands.append(d_val)
                child_nodes.append(self.graph.get_node(child_idx))
            
            total_demand = sum(branch_demands)
            node_split_config = {}
            if total_demand > 1e-3:
                for i, child_name in enumerate(child_nodes):
                    ratio = branch_demands[i] / total_demand
                    node_split_config[child_name] = np.clip(ratio, 0.05, 0.95)
            else:
                base_ratio = 1.0 / len(child_nodes)
                for child_name in child_nodes:
                    node_split_config[child_name] = base_ratio
            
            heuristic_splits[b_node_str] = node_split_config
        return heuristic_splits

    def solve(self, tspan, y0, **kwargs):
        return solve_ivp(lambda t, y: self.compute_system_dynamics(t, y), tspan, y0, **kwargs)

    def compute_system_dynamics(self, t, T_cells):
        dy_dt = np.zeros_like(T_cells)
        pipe_m_flows = self._compute_mass_flows(t)
        
        cell_m_flows = pipe_m_flows[self.cell_to_pipe_idx]
        cell_velocities = cell_m_flows / (self.rho * self.all_areas)
        
        node_temps = self._solve_nodes_temperature(T_cells, pipe_m_flows, t)
        
        if len(self.consumer_indices) > 0:
            node_temps = self._apply_power_consumption_fast(t, node_temps, pipe_m_flows)
            
        ext = self.ext_temp_val_or_func(t) if callable(self.ext_temp_val_or_func) else self.ext_temp_val_or_func
        
        dy_dt[self.ids_internal_curr] = -(cell_velocities[self.ids_internal_curr] / self.all_dx[self.ids_internal_curr]) * \
                                        (T_cells[self.ids_internal_curr] - T_cells[self.ids_internal_up])
        
        T_start_inlet = node_temps[self.pipe_u_idxs]
        dy_dt[self.idx_pipe_start] = -(cell_velocities[self.idx_pipe_start] / self.all_dx[self.idx_pipe_start]) * \
                                     (T_cells[self.idx_pipe_start] - T_start_inlet)
        
        dy_dt -= self.all_loss_factors * (T_cells - ext)
        return dy_dt

    def _compute_mass_flows(self, t):
        mass_flows = np.zeros(len(self.pipe_configs))
        node_mass_in = np.zeros(self.graph.n_nodes)

        inlet_val = self.inlet_mass_flow(t) if callable(self.inlet_mass_flow) else self.inlet_mass_flow
        node_mass_in[self.graph.get_id(self.graph.inlet_node)] = inlet_val

        for u_idx in self.graph.topo_indices:
            m_in = node_mass_in[u_idx]
            if m_in <= 1e-9: continue
            
            children = self.graph.int_adjacency[u_idx]
            for v_idx, pipe_idx in children:
                split = self.split_matrix[u_idx, v_idx]
                m_out = m_in * split
                mass_flows[pipe_idx] = m_out
                node_mass_in[v_idx] += m_out
        return mass_flows

    def _solve_nodes_temperature(self, T_cells, pipe_m_flows, t):
        node_mass = np.zeros(self.graph.n_nodes)
        node_enthalpy = np.zeros(self.graph.n_nodes)

        mask = pipe_m_flows > 1e-9
        if np.any(mask):
            target_nodes = self.pipe_v_idxs[mask]
            flows = pipe_m_flows[mask]
            temps = T_cells[self.idx_pipe_end[mask]]
            np.add.at(node_mass, target_nodes, flows)
            np.add.at(node_enthalpy, target_nodes, flows * temps)

        with np.errstate(divide='ignore', invalid='ignore'):
            node_temps = node_enthalpy / node_mass
        node_temps[np.isnan(node_temps)] = self.t_min_return
        
        T_src = self.inlet_temp(t) if callable(self.inlet_temp) else self.inlet_temp
        node_temps[self.graph.get_id(self.graph.inlet_node)] = T_src
        return node_temps

    def _apply_power_consumption_fast(self, t, node_temps, pipe_mass_flows):
        m_in_cons = np.zeros(len(self.consumer_indices))
        for i, idx in enumerate(self.consumer_indices):
            total_m = 0.0
            for _, pipe_idx in self.graph.int_parent_adjacency[idx]:
                total_m += pipe_mass_flows[pipe_idx]
            m_in_cons[i] = total_m

        p_targets = np.array([self.node_power_funcs[nid](t) for nid in self.consumer_ids_str])
        T_in = node_temps[self.consumer_indices]
        delta_T_avail = np.maximum(T_in - self.t_min_return, 0.0)
        p_max = m_in_cons * self.cp * delta_T_avail
        
        p_effective = np.minimum(p_targets, p_max)
        with np.errstate(divide='ignore', invalid='ignore'):
            delta_T = p_effective / (m_in_cons * self.cp)
        delta_T[np.isnan(delta_T)] = 0.0
        
        node_temps[self.consumer_indices] = np.maximum(T_in - delta_T, self.t_min_return)
        return node_temps

    def get_instant_metrics(self, t, state, pipe_flows=None, node_temps=None):
        if pipe_flows is None:
            pipe_flows = self._compute_mass_flows(t)
        if node_temps is None:
            node_temps = self._solve_nodes_temperature(state, pipe_flows, t)
        
        node_temps_after = self._apply_power_consumption_fast(t, node_temps.copy(), pipe_flows)
        
        m_in_cons = np.zeros(len(self.consumer_indices))
        for i, idx in enumerate(self.consumer_indices):
            m = 0.0
            for _, pipe_idx in self.graph.int_parent_adjacency[idx]:
                m += pipe_flows[pipe_idx]
            m_in_cons[i] = m

        p_targets = np.array([self.node_power_funcs[nid](t) for nid in self.consumer_ids_str])
        T_in = node_temps[self.consumer_indices]
        p_max = m_in_cons * self.cp * np.maximum(T_in - self.t_min_return, 0.0)
        p_supplied_vec = np.minimum(p_targets, p_max)
        
        total_demand = np.sum(p_targets)
        total_supplied = np.sum(p_supplied_vec)
        
        wasted = 0.0
        for idx in self.terminal_indices:
            m_in = 0.0
            for _, pipe_idx in self.graph.int_parent_adjacency[idx]:
                m_in += pipe_flows[pipe_idx]
            if m_in > 1e-9:
                T_out = node_temps_after[idx]
                if T_out > self.t_min_return:
                    wasted += m_in * self.cp * (T_out - self.t_min_return)

        inlet_idx = self.graph.get_id(self.graph.inlet_node)
        m_pumped = 0.0
        for _, pipe_idx in self.graph.int_adjacency[inlet_idx]:
            m_pumped += pipe_flows[pipe_idx]
        p_pump = 1000.0 * m_pumped 

        return {
            "demand": total_demand,
            "supplied": total_supplied,
            "wasted": wasted,
            "pump_power": p_pump,
            "node_temperatures": node_temps,
            "node_temperatures_after": node_temps_after,
            "mass_flows": pipe_flows
        }