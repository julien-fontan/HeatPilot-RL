import os
import numpy as np
import matplotlib.pyplot as plt
from district_heating_model import PipeConfig, DistrictHeatingNetwork
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

# Dossier pour stocker tous les plots
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def run_simulation():
    seed = GLOBAL_SEED
    edges = EDGES
    props = PHYSICAL_PROPS
    dx = SIMULATION_PARAMS["dx"]
    t_max_day = SIMULATION_PARAMS["t_max_day"]
    dt = TRAINING_PARAMS["dt"]

    warmup_duration = SIMULATION_PARAMS["warmup"]
    warmup_duration = 0

    graph = Graph(edges)
    consumer_nodes = graph.get_consumer_nodes()

    # 1. Génération des configs de tuyaux
    lengths, diameters, n_segments, h_vals = PipeConfig.generate_parameters(
        edges=edges,
        dx=dx,
        seed=seed,
    )

    pipe_configs = []
    for i, (u_node, v_node) in enumerate(edges):
        pc = PipeConfig(
            nodes=(u_node, v_node),
            length=float(lengths[i]),
            diameter=float(diameters[i]),
            dx=dx,
            rho=props["rho"],
            cp=props["cp"],
            heat_loss_coeff=float(h_vals[i]),
            thermal_conductivity=props["thermal_conductivity"],
            external_temp=props["external_temp"],
        )
        pipe_configs.append(pc)

    # Entrées constantes pour ce test
    inlet_temp = 70.0
    inlet_mass_flow = 12.0

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
        t_min_return=SIMULATION_PARAMS["min_return_temp"]
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

    # --- Reconstruction des métriques ---
    # Le nouveau moteur a une méthode dédiée pour ça
    metrics = network.reconstruct_metrics(sol.t, sol.y)
    
    p_demand_tot = metrics["total_demand"]
    p_supplied_tot = metrics["total_supplied"]
    
    # Calcul Boiler Power (post-traitement simple car débit/temp constants ici)
    # Si variables, il faudrait réévaluer inlet_mass_flow(t) * cp * (inlet_temp(t) - T_ret)
    # Ici simple approximation : P = m * cp * (T_in - T_min_ret)
    p_boiler = inlet_mass_flow * props["cp"] * (inlet_temp - SIMULATION_PARAMS["min_return_temp"])
    p_boiler_vec = np.full_like(sol.t, p_boiler)

    # --- Plotting ---
    time_hours = sol.t / 3600.0

    # Lissage
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w

    window_size = int(5*60.0 / dt)
    if window_size < 1: window_size = 1
    
    p_demand_smooth = moving_average(p_demand_tot, window_size)
    p_supplied_smooth = moving_average(p_supplied_tot, window_size)
    p_boiler_smooth = moving_average(p_boiler_vec, window_size)
    time_smooth = time_hours[window_size-1:]

    plt.figure(figsize=(8, 4))
    plt.plot(time_smooth, p_demand_smooth / 1e3, label="Demand total")
    plt.plot(time_smooth, p_supplied_smooth / 1e3, label="Supplied total")
    plt.plot(time_smooth, p_boiler_smooth / 1e3, label="Boiler power")
    plt.xlabel("Time (h)")
    plt.ylabel("Power (kW)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(PLOTS_DIR, "power_balance_simulation.svg")
    plt.savefig(plot_path, transparent=True)
    print(f"Figure sauvegardée dans {plot_path}")
    plt.show()

if __name__ == "__main__":
    run_simulation()