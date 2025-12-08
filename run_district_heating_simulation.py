import numpy as np
import matplotlib.pyplot as plt
from district_heating_model import Pipe, DistrictHeatingNetwork
from utils import generate_step_function, generate_smooth_profile
from graph_utils import Graph
from config import (
    EDGES,
    PHYSICAL_PROPS,
    SIMULATION_PARAMS,
    GLOBAL_SEED,
    POWER_PROFILE_CONFIG,
    MIN_RETURN_TEMP,
    PIPE_GENERATION,
    RL_TRAINING,   # <-- importer le dt RL
)

def run_simulation():
    edges = EDGES
    props = PHYSICAL_PROPS
    dx = SIMULATION_PARAMS["dx"]
    t_max_day = SIMULATION_PARAMS["t_max_day"]
    dt = RL_TRAINING["dt"]       # pas temporel utilisé pour l’échantillonnage des courbes
    seed = GLOBAL_SEED  # même graine que le RL, pour cohérence globale

    # durée de pré-chauffe (warmup) pour dépasser le transitoire initial
    warmup_duration = SIMULATION_PARAMS["warmup"]

    # Graphe pour connaître les noeuds consommateurs
    graph = Graph(edges)
    consumer_nodes = graph.get_consumer_nodes()

    # Génération reproductible des propriétés des conduites, paramètres depuis config via defaults
    lengths, diameters, n_segments, h_vals = Pipe.generate_parameters(
        edges=edges,
        dx=dx,
        seed=seed,
        # n_segments_min=PIPE_GENERATION["n_segments_min"],   # facultatif
        # n_segments_max=PIPE_GENERATION["n_segments_max"],
        # diameter_min=PIPE_GENERATION["diameter_min"],
        # diameter_max=PIPE_GENERATION["diameter_max"],
        # h_min=PIPE_GENERATION["h_min"],
        # h_max=PIPE_GENERATION["h_max"],
    )

    pipes_list = []
    for i, (u_node, v_node) in enumerate(edges):
        p = Pipe(
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
        pipes_list.append(p)

    # Température source en entrée
    inlet_temp = 100.0  # °C constante
    # inlet_temp = generate_step_function(t_max_day, 900.0, 70.0, 90.0, seed=seed)   # °C

    # Débit massique en entrée
    inlet_mass_flow = 15.0  # kg/s constant
    # inlet_mass_flow = generate_step_function(t_max_day, 900.0, 10.0, 20.0, seed=seed)  # kg/s

    # Profils de puissance demandée par les noeuds consommateurs
    rng = np.random.default_rng()  # pas de seed ici, profils différents à chaque exécution
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
    
    # # A décommenter si l'on veut une même puissance consommée constante sur tous les noeuds
    # constant_power = POWER_PROFILE_CONFIG["p_max"]  # par ex. une valeur unique
    # node_power_funcs = {node: (lambda t, p=constant_power: p) for node in consumer_nodes}
    
    network = DistrictHeatingNetwork(
        graph=graph,
        pipes=pipes_list,
        inlet_mass_flow=inlet_mass_flow,
        inlet_temp=inlet_temp,
        rho=props["rho"],
        cp=props["cp"],
        node_power_funcs=node_power_funcs,
    )

    # --- État initial froid ---
    # Température : MIN_RETURN_TEMP dans toutes les cellules de toutes les conduites
    y0 = np.full(network.total_cells, MIN_RETURN_TEMP)

    # --- Phase 1 : warmup (aucun enregistrement utilisé ensuite) ---
    print("Warmup du réseau...")
    sol_warmup = network.solve(
        (0.0, warmup_duration),
        y0,
        method="BDF",
        rtol=SIMULATION_PARAMS["rtol"],
        atol=SIMULATION_PARAMS["atol"],
    )
    y_warm = sol_warmup.y[:, -1]  # état du réseau après warmup

    # --- Phase 2 : simulation principale à partir de l'état chaud ---
    print("Début de la simulation principale...")
    # points d’échantillonnage toutes dt secondes (ou un multiple si tu veux plus grossier)
    t_eval_points = np.arange(warmup_duration, t_max_day + dt, dt)
    sol = network.solve(
        (warmup_duration, t_max_day),
        y_warm,
        method="BDF",
        rtol=SIMULATION_PARAMS["rtol"],
        atol=SIMULATION_PARAMS["atol"],
        t_eval=t_eval_points,
    )
    print("Simulation terminée.")

    # --- Puissances totales sur la phase principale (côté consommateurs) ---
    p_demand_tot = network.get_total_power_demand(sol.t)      # W
    p_supplied_tot = network.get_total_power_supplied(sol.t)  # W

    # --- Puissances chaudière + pompe (côté production) via getters ---
    p_boiler = network.get_boiler_power(sol.t)  # W
    p_pump = network.get_pump_power(sol.t)      # W

    # --- Figures ---
    time_hours = sol.t / 3600.0

    # 1) Bilan demande vs fournie aux consommateurs
    plt.figure(figsize=(8, 4))
    plt.plot(time_hours, p_demand_tot / 1e3, label="P_demand_tot (kW)")
    plt.plot(time_hours, p_supplied_tot / 1e3, label="P_supplied_tot (kW)")
    plt.plot(time_hours, p_boiler / 1e3, label="P_boiler (kW)")
    plt.xlabel("Temps (h)")
    plt.ylabel("Puissance (kW)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # plt.savefig("power_balance_consumers.png", dpi=150)

    # # --- Visualisation 1D (déjà existante) ---
    # pipe_idx = 0
    # indices = network.pipe_slices[pipe_idx]
    # T_pipe_output = sol.y[indices, :][-1, :]
    # np.savetxt("pipe_0_output.csv", sol.y[indices, :], delimiter=",")

    # # --- Export des débits massiques et vitesses (cas constant) ---
    # # Ici inlet_mass_flow est constant → get_flows_and_velocities renvoie deux vecteurs 1D
    # mass_flows, velocities = network.get_flows_and_velocities()
    # np.savetxt("mass_flows_final.csv", mass_flows, delimiter=",")
    # np.savetxt("velocities_final.csv", velocities, delimiter=",")

    # # --- Visualisation du graphe avec températures et mass flows ---
    # t_final = tspan[1]
    # state_final = sol.y[:, -1]
    # fig_graph, ax_graph = plot_temperature_graph(network, state_final, t_final)
    # fig_graph.savefig("network_temperature_graph.png", dpi=150)
    # # fig_graph.show()  # à activer si tu veux l'afficher interactif

if __name__ == "__main__":
    run_simulation()
