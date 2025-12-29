import os
import numpy as np
from district_heating_model import PipeConfig, DistrictHeatingNetwork, create_pipe_configs
from utils import generate_smooth_profile
from graph_utils import Graph
from config import (
    GLOBAL_SEED,
    EDGES,
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

def run_simulation(node_splits=None):

    seed = GLOBAL_SEED
    edges = EDGES
    props = PHYSICAL_PROPS
    dx = SIMULATION_PARAMS["dx"]
    t_max_day = SIMULATION_PARAMS["t_max_day"]
    dt = TRAINING_PARAMS["dt"]
    warmup_duration = 0

    graph = Graph(edges)
    consumer_nodes = graph.get_consumer_nodes()
    # ...existing code...

    node_splits={3: {7: 0.3, 4: 0.7}, 5: {6: 0.3, 9: 0.7}}

    # 1. Génération des configs de tuyaux
    lengths, diameters, n_segments, h_vals = PipeConfig.generate_parameters(
        edges=edges,
        dx=dx,
        seed=seed,
    )

    pipe_configs = create_pipe_configs(edges, lengths, diameters, h_vals, dx, props)

    # Entrées constantes pour ce test
    inlet_temp = 75.0
    inlet_mass_flow = 15.0

    # Profils de puissance
    rng = np.random.default_rng()
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
    t_eval_points = np.arange(warmup_duration, t_max_day + dt, dt)
    sol = network.solve(
        (warmup_duration, t_max_day),
        y_warm,
        method="RK45",
        rtol=SIMULATION_PARAMS["rtol"],
        atol=SIMULATION_PARAMS["atol"],
        t_eval=t_eval_points,
    )
    print("Simulation terminée.")


    # Préparation des données pour la visualisation factorisée
    # On suppose que metrics contient les clés nécessaires pour la visualisation
    # Pour compatibilité, on ajoute les clés attendues par la visualisation
    metrics = network.reconstruct_metrics(sol.t, sol.y)

    data = dict(metrics)
    # Remap keys for visualizer compatibility
    data["demand_total"] = metrics["total_demand"]
    data["supplied_total"] = metrics["total_supplied"]
    data["wasted_total"] = metrics["total_wasted"]
    data["time"] = sol.t
    data["T_source"] = np.full_like(sol.t, inlet_temp)
    data["m_source"] = np.full_like(sol.t, inlet_mass_flow)

    data["node_ids"] = np.array(consumer_nodes)

    # Ajout des signaux individuels par nœud pour la visualisation détaillée
    parents_map = graph.get_parent_nodes()
    cp = props["cp"]
    min_ret_temp = SIMULATION_PARAMS["min_return_temp"]
    for nid in consumer_nodes:
        p_dem_list = []
        p_sup_list = []
        m_in_list = []
        T_in_list = []
        for k in range(len(data["time"])):
            t = data["time"][k]
            # Recalcule les débits et températures nodales à chaque pas de temps
            T_state = sol.y[:, k]
            m_flows = network._compute_mass_flows(t)
            node_temps = network._solve_nodes_temperature(T_state, m_flows, t)
            node_idx = graph.get_id_from_node(nid)
            m_in = sum(m_flows[graph.edges[(p_node, nid)]["pipe_index"]] for p_node in parents_map.get(nid, []))
            T_in = node_temps[node_idx]
            p_dem = node_power_funcs[nid](t)
            delta_T_avail = max(T_in - min_ret_temp, 0.0)
            p_phys_max = m_in * cp * delta_T_avail
            p_sup = min(p_dem, p_phys_max)
            p_dem_list.append(p_dem)
            p_sup_list.append(p_sup)
            m_in_list.append(m_in)
            T_in_list.append(T_in)
        data[f"node_{nid}_p_dem"] = np.array(p_dem_list)
        data[f"node_{nid}_p_sup"] = np.array(p_sup_list)
        data[f"node_{nid}_m_in"] = np.array(m_in_list)
        data[f"node_{nid}_T_in"] = np.array(T_in_list)

    visualizer = DistrictHeatingVisualizer(PLOTS_DIR)
    visualizer.plot_dashboard_general(data, title_suffix="simulation")
    visualizer.plot_dashboard_nodes_2cols(data, title_suffix="simulation")

    # visualizer.plot_dashboard_nodes_2cols(data, title_suffix="simulation")  # À activer si les données sont prêtes

if __name__ == "__main__":
    run_simulation()