import os
import numpy as np
import argparse
from district_heating_model import PipeConfig, DistrictHeatingNetwork, create_pipe_configs
from utils import generate_smooth_profile
from graph_utils import Graph
from config import GLOBAL_SEED, EDGES, PHYSICAL_PROPS, SIMULATION_PARAMS, POWER_PROFILE_CONFIG, TRAINING_PARAMS, DEFAULT_NODE_SPLITS
from graph_visualization import DistrictHeatingVisualizer

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def run_simulation(node_splits=None, auto_valves: bool = False):
    graph = Graph(EDGES)
    
    lengths, diameters, _, h_vals = PipeConfig.generate_parameters(EDGES, SIMULATION_PARAMS["dx"], GLOBAL_SEED)
    pipe_configs = create_pipe_configs(EDGES, lengths, diameters, h_vals, SIMULATION_PARAMS["dx"], PHYSICAL_PROPS)

    rng = np.random.default_rng(GLOBAL_SEED)
    node_power_funcs = {}
    for node in graph.consumer_nodes:
        node_power_funcs[node] = generate_smooth_profile(
            t_end=SIMULATION_PARAMS["t_max_day"],
            step_time=POWER_PROFILE_CONFIG["step_time"],
            min_val=POWER_PROFILE_CONFIG["p_min"],
            max_val=POWER_PROFILE_CONFIG["p_max"],
            seed=rng.integers(0, 1_000_000)
        )
    
    network = DistrictHeatingNetwork(
        graph=graph, pipe_configs=pipe_configs,
        inlet_mass_flow=15.0, inlet_temp=72.0,
        rho=PHYSICAL_PROPS["rho"], cp=PHYSICAL_PROPS["cp"],
        node_power_funcs=node_power_funcs,
        t_min_return=SIMULATION_PARAMS["min_return_temp"],
        node_splits=node_splits or DEFAULT_NODE_SPLITS
    )

    y = np.full(network.total_cells, SIMULATION_PARAMS["min_return_temp"])
    
    print("Warmup...")
    network.inlet_temp = 72.0
    network.solve((0, SIMULATION_PARAMS["warmup_duration"]), y, method="RK45")
    
    print("Simulation...")
    t, dt = 0.0, TRAINING_PARAMS["dt"]
    n_steps = int(SIMULATION_PARAMS["t_max_day"] / dt)
    
    hist = {"time": [], "dem": [], "sup": [], "wast": [], "splits": {b: [] for b in graph.branching_nodes}}
    node_hist = {n: {"dem": [], "sup": [], "m": [], "T": []} for n in graph.consumer_nodes}

    for _ in range(n_steps):
        t_next = t + dt
        
        if auto_valves:
            splits = network.compute_heuristic_splits(t)
            network.set_node_splits(splits)
            for b in graph.branching_nodes:
                if b in splits:
                    child0 = graph.child_nodes[b][0]
                    hist["splits"][b].append(splits[b].get(child0, 0.5))
                else: hist["splits"][b].append(0.5)
        else:
            for b in graph.branching_nodes:
                child0 = graph.child_nodes[b][0]
                val = (node_splits or DEFAULT_NODE_SPLITS).get(b, {}).get(child0, 0.5)
                hist["splits"][b].append(val)

        sol = network.solve((t, t_next), y, method="RK45")
        y = sol.y[:, -1]
        
        metrics = network.get_instant_metrics(t, y)
        
        hist["time"].append(t)
        hist["dem"].append(metrics["demand"])
        hist["sup"].append(metrics["supplied"])
        hist["wast"].append(metrics["wasted"])
        
        for nid in graph.consumer_nodes:
            idx = graph.get_id(nid)
            node_hist[nid]["dem"].append(node_power_funcs[nid](t))
            
            m_in = 0.0
            for _, pidx in graph.int_parent_adjacency[idx]: m_in += metrics["mass_flows"][pidx]
            avail = m_in * PHYSICAL_PROPS["cp"] * max(metrics["node_temperatures"][idx] - SIMULATION_PARAMS["min_return_temp"], 0)
            
            node_hist[nid]["sup"].append(min(node_power_funcs[nid](t), avail))
            node_hist[nid]["m"].append(m_in)
            node_hist[nid]["T"].append(metrics["node_temperatures"][idx])

        t = t_next

    data = {
        "time": np.array(hist["time"]),
        "demand_total": np.array(hist["dem"]),
        "supplied_total": np.array(hist["sup"]),
        "wasted_total": np.array(hist["wast"]),
        "T_source": np.full(len(hist["time"]), 72.0),
        "m_source": np.full(len(hist["time"]), 15.0),
        "node_ids": np.array(graph.consumer_nodes)
    }
    for b, vals in hist["splits"].items(): data[f"split_node_{b}"] = np.array(vals)
    for nid in graph.consumer_nodes:
        for k in ["dem", "sup", "m", "T"]:
            key = f"p_{k}" if k in ["dem", "sup"] else (f"{k}_in")
            data[f"node_{nid}_{key}"] = np.array(node_hist[nid][k])

    viz = DistrictHeatingVisualizer(PLOTS_DIR)
    suffix = "auto" if auto_valves else "fixed"
    viz.plot_dashboard_general(data, title_suffix=suffix)
    viz.plot_dashboard_nodes_2cols(data, title_suffix=suffix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-valves", action="store_true")
    args = parser.parse_args()
    run_simulation(auto_valves=args.auto_valves)