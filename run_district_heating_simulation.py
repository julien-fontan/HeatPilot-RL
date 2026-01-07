import os
import numpy as np
import argparse
from district_heating_model import PipeConfig, DistrictHeatingNetwork, create_pipe_configs
from utils import generate_smooth_profile
from graph_utils import Graph
from config import (
    GLOBAL_SEED,
    EDGES,
    DEFAULT_NODE_SPLITS,
    PHYSICAL_PROPS,
    SIMULATION_PARAMS,
    POWER_PROFILE_CONFIG,
    TRAINING_PARAMS
)
from graph_visualization import DistrictHeatingVisualizer

# Dossier pour stocker tous les plots
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def _compute_ideal_node_splits(graph: Graph, downstream_consumers_map, node_power_funcs, t: float):
    """Heuristique: split = demande aval branche A / (A+B) pour chaque noeud de branchement."""
    all_children = graph.get_child_nodes()
    branching_nodes = graph.get_branching_nodes()
    branching_map = {n: all_children[n] for n in branching_nodes}

    node_splits = {}
    for b_node in branching_nodes:
        children = branching_map[b_node]
        if len(children) < 2:
            continue

        child_1, child_2 = children[0], children[1]
        demand_1 = sum(node_power_funcs[c](t) for c in downstream_consumers_map[b_node][child_1])
        demand_2 = sum(node_power_funcs[c](t) for c in downstream_consumers_map[b_node][child_2])
        total = demand_1 + demand_2

        if total > 1e-3:
            ratio = demand_1 / total
        else:
            ratio = 0.5

        ratio = float(np.clip(ratio, 0.05, 0.95))
        node_splits[b_node] = {child_1: ratio, child_2: 1.0 - ratio}

    return node_splits


def run_simulation(node_splits=None, auto_valves: bool = False):

    seed = GLOBAL_SEED
    edges = EDGES
    props = PHYSICAL_PROPS
    dx = SIMULATION_PARAMS["dx"]
    t_max_day = SIMULATION_PARAMS["t_max_day"]
    dt = TRAINING_PARAMS["dt"]
    warmup_duration = 0

    graph = Graph(edges)
    consumer_nodes = graph.get_consumer_nodes()

    if node_splits is None:
        node_splits = DEFAULT_NODE_SPLITS

    # 1. Génération des configs de tuyaux
    lengths, diameters, n_segments, h_vals = PipeConfig.generate_parameters(
        edges=edges,
        dx=dx,
        seed=seed,
    )

    pipe_configs = create_pipe_configs(edges, lengths, diameters, h_vals, dx, props)

    # Entrées constantes pour ce test
    inlet_temp = 72.0
    inlet_mass_flow = 15.0

    # Profils de puissance
    rng = np.random.default_rng(seed)
    smooth_factor = POWER_PROFILE_CONFIG.get("smooth_factor", 1.0)
    node_power_funcs = {}
    for node in consumer_nodes:
        node_power_funcs[node] = generate_smooth_profile(
            t_end=t_max_day,
            step_time=POWER_PROFILE_CONFIG["step_time"],
            min_val=POWER_PROFILE_CONFIG["p_min"],
            max_val=POWER_PROFILE_CONFIG["p_max"],
            seed=rng.integers(0, 1_000_000),
            smooth_factor=smooth_factor,
        )
    
    # 2. Construction Réseau Vectorisé
    network = DistrictHeatingNetwork(
        graph=graph,
        pipe_configs=pipe_configs,
        inlet_mass_flow=inlet_mass_flow,
        inlet_temp=inlet_temp,
        rho=props["rho"],
        cp=props["cp"],
        node_power_funcs=node_power_funcs,
        t_min_return=SIMULATION_PARAMS["min_return_temp"],
        node_splits=node_splits
    )

    # État initial
    y0 = np.full(network.total_cells, SIMULATION_PARAMS["min_return_temp"])

    print("Warmup du réseau...")
    sol_warmup = network.solve(
        (0.0, warmup_duration),
        y0,
        method="RK45", # RK45 souvent plus robuste que BDF pour ce type de système advectif
        rtol=SIMULATION_PARAMS["rtol"],
        atol=SIMULATION_PARAMS["atol"],
    )
    y_warm = sol_warmup.y[:, -1]

    print("Début de la simulation principale...")

    # NOTE: si on change les vannes au cours du temps, on ne peut pas utiliser un seul solve_ivp
    # puis reconstruire les métriques, car les splits ne sont pas une fonction de t dans le moteur.
    # On intègre donc pas-à-pas avec mise à jour (optionnelle) des vannes.
    downstream_consumers_map = graph.get_downstream_consumers_map()
    branching_nodes = graph.get_branching_nodes()
    all_children = graph.get_child_nodes()
    branching_map = {n: all_children[n] for n in branching_nodes}

    terminal_nodes = graph.get_terminal_nodes()
    parents_map = graph.get_parent_nodes()
    cp = props["cp"]
    min_ret_temp = SIMULATION_PARAMS["min_return_temp"]

    times = []
    demand_total = []
    supplied_total = []
    wasted_total = []
    T_source = []
    m_source = []

    # Signaux par noeud
    node_signals = {nid: {"p_dem": [], "p_sup": [], "m_in": [], "T_in": []} for nid in consumer_nodes}
    # Suivi splits (ratio vers le 1er enfant)
    split_hist = {b: [] for b in branching_nodes}

    t = float(warmup_duration)
    y = y_warm

    n_steps = int(np.ceil((t_max_day - warmup_duration) / dt))
    for _ in range(n_steps):
        t_next = min(t + dt, t_max_day)

        if auto_valves:
            node_splits_t = _compute_ideal_node_splits(graph, downstream_consumers_map, node_power_funcs, t)
            network.set_node_splits(node_splits_t)
        else:
            node_splits_t = node_splits

        # Historique splits (si noeud de branchement avec 2 enfants)
        for b in branching_nodes:
            children = branching_map[b]
            if len(children) < 2:
                continue
            child1 = children[0]
            frac = node_splits_t.get(b, {}).get(child1, 0.5)
            split_hist[b].append(frac)

        # Intégration sur [t, t_next]
        sol_step = network.solve(
            (t, t_next),
            y,
            method="RK45",
            rtol=SIMULATION_PARAMS["rtol"],
            atol=SIMULATION_PARAMS["atol"],
            t_eval=[t_next],
        )
        y = sol_step.y[:, -1]
        t = t_next

        # Métriques au temps courant (comme env.step())
        pipe_flows = network._compute_mass_flows(t)
        node_temps = network._solve_nodes_temperature(y, pipe_flows, t)
        node_temps_after = network._apply_node_power_consumption(t, node_temps, pipe_flows)

        step_demand = 0.0
        step_supplied = 0.0
        for nid in consumer_nodes:
            p_dem = float(node_power_funcs[nid](t))
            step_demand += p_dem

            m_in = 0.0
            for p_node in parents_map.get(nid, []):
                m_in += pipe_flows[graph.edges[(p_node, nid)]["pipe_index"]]

            node_idx = graph.get_id_from_node(nid)
            T_in = float(node_temps[node_idx])

            if m_in > 1e-9:
                delta_T_avail = max(T_in - min_ret_temp, 0.0)
                p_phys_max = m_in * cp * delta_T_avail
                p_sup = min(p_dem, p_phys_max)
            else:
                p_sup = 0.0

            node_signals[nid]["p_dem"].append(p_dem)
            node_signals[nid]["p_sup"].append(p_sup)
            node_signals[nid]["m_in"].append(m_in)
            node_signals[nid]["T_in"].append(T_in)

            step_supplied += p_sup

        # Wasted (terminaux)
        step_wasted = 0.0
        for nid in terminal_nodes:
            m_in = 0.0
            for p_node in parents_map.get(nid, []):
                m_in += pipe_flows[graph.edges[(p_node, nid)]["pipe_index"]]

            if m_in > 1e-9:
                idx = graph.get_id_from_node(nid)
                T_after = float(node_temps_after[idx])
                step_wasted += m_in * cp * max(T_after - min_ret_temp, 0.0)

        times.append(t)
        demand_total.append(step_demand)
        supplied_total.append(step_supplied)
        wasted_total.append(step_wasted)
        T_source.append(inlet_temp)
        m_source.append(inlet_mass_flow)

        if t >= t_max_day:
            break

    print("Simulation terminée.")

    data = {
        "time": np.array(times),
        "demand_total": np.array(demand_total),
        "supplied_total": np.array(supplied_total),
        "wasted_total": np.array(wasted_total),
        "T_source": np.array(T_source),
        "m_source": np.array(m_source),
        "node_ids": np.array(consumer_nodes),
    }

    for nid in consumer_nodes:
        data[f"node_{nid}_p_dem"] = np.array(node_signals[nid]["p_dem"])
        data[f"node_{nid}_p_sup"] = np.array(node_signals[nid]["p_sup"])
        data[f"node_{nid}_m_in"] = np.array(node_signals[nid]["m_in"])
        data[f"node_{nid}_T_in"] = np.array(node_signals[nid]["T_in"])

    for b in branching_nodes:
        data[f"split_node_{b}"] = np.array(split_hist[b])

    visualizer = DistrictHeatingVisualizer(PLOTS_DIR)
    suffix = "simulation_auto_valves" if auto_valves else "simulation_fixed_valves"
    visualizer.plot_dashboard_general(data, title_suffix=suffix)
    visualizer.plot_dashboard_nodes_2cols(data, title_suffix=suffix)

    # visualizer.plot_dashboard_nodes_2cols(data, title_suffix="simulation")  # À activer si les données sont prêtes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--auto-valves",
        action="store_true",
        help="Active un contrôle automatique des vannes (splits recalculés à chaque pas).",
    )
    args = parser.parse_args()

    run_simulation(auto_valves=args.auto_valves)