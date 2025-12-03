import numpy as np
import matplotlib.pyplot as plt
from heat_network import Pipe, DistrictHeatingNetwork
from utils import generate_step_function, PhysicalProperties
# from graph_viz import plot_temperature_graph

# --- Topologie Globale ---
# Définie ici pour être partagée entre la simulation et l'entraînement RL
EDGES = [
    (1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),
    (8,20),
    (7,18),(18,19),
    (6,15),(15,16),
    (5,13),(13,14),
    (4,23),(23,24),
    (3,10),(10,11)
]

def run_simulation():
    # --- Configuration ---
    props = PhysicalProperties(
        rho=1000.0,
        cp=4182.0,
        thermal_conductivity=0.0,
        external_temp=10.0,
        heat_loss_coeff=1.5 # Valeur moyenne par défaut
    )
    dx = 10.0
    tspan = (0.0, 20000.0)
    seed = 42

    power = 2e6  # watts

    # Utilisation de la constante globale
    edges = EDGES
    n_pipes = len(edges)

    rng = np.random.default_rng(seed)
    
    # Génération aléatoire des propriétés physiques
    n_segments = rng.integers(20, 61, size=n_pipes)
    lengths = dx * n_segments
    diameters = rng.uniform(0.15, 0.35, size=n_pipes)
    # heat_coeffs = rng.uniform(0.90, 1.75, size=n_pipes) # (h) -> Supprimé car factorisé (ou à gérer différemment si hétérogène)

    pipes_list = []
    for i, (u_node, v_node) in enumerate(edges):
        # Création de la conduite
        p = Pipe(
            nodes=(u_node, v_node),
            length=float(lengths[i]),
            diameter=float(diameters[i]),
            # heat_loss_coeff supprimé ici
            dx=dx,
            props=props
        )
        pipes_list.append(p)

    # --- Modèle de débit très simplifié ---
    inlet_node_id = 1           # premier noeud
    inlet_mass_flow = 10.0  # kg/s direct, sans passer par une vitesse

    # Fractions de répartition aux noeuds ramifiés (toujours un simple dict utilisateur)
    node_splits = {
        3: {4: 1, 10: 0},
        4: {5: 0.8, 23: 0.2},
        5: {6: 0.6, 13: 0.4},
        6: {7: 0.7, 15: 0.3},
        7: {8: 0.7, 18: 0.3},
        8: {9: 0.7, 20: 0.3},
    }

    # Création du réseau avec config débit intégrée
    network = DistrictHeatingNetwork(
        graph=None, # Sera créé automatiquement
        pipes=pipes_list,
        props=props,
        inlet_node_id=inlet_node_id,
        inlet_mass_flow=inlet_mass_flow,
        node_splits=node_splits,
    )

    # Conditions aux limites (Source au noeud 1)
    temp_source_func = generate_step_function(tspan[1], 900.0, 70.0, 90.0, seed=seed)
    boundary_conditions = {1: temp_source_func}

    # État initial (70°C partout)
    y0 = np.full(network.total_cells, 70.0)

    print("Début de la simulation...")
    # t_eval force le solveur à interpoler les résultats sur une grille de temps régulière (ex: toutes les 10s)
    t_eval_points = np.arange(tspan[0], tspan[1], 10.0)
    sol = network.solve(
        tspan,
        y0,
        boundary_conditions,
        method="BDF",
        rtol=1e-4,
        atol=1e-8,
        t_eval=t_eval_points,
    )
    print("Simulation terminée.")

    # --- Visualisation 1D (déjà existante) ---
    pipe_idx = 0
    indices = network.pipe_slices[pipe_idx]
    T_pipe_output = sol.y[indices, :][-1, :]
    np.savetxt("pipe_0_output.csv", sol.y[indices, :], delimiter=",")

    # --- Export des débits massiques et vitesses (cas constant) ---
    # Ici inlet_mass_flow est constant → get_flows_and_velocities renvoie deux vecteurs 1D
    mass_flows, velocities = network.get_flows_and_velocities()
    np.savetxt("mass_flows_final.csv", mass_flows, delimiter=",")
    np.savetxt("velocities_final.csv", velocities, delimiter=",")

    # # --- Visualisation du graphe avec températures et mass flows ---
    # t_final = tspan[1]
    # state_final = sol.y[:, -1]
    # fig_graph, ax_graph = plot_temperature_graph(network, state_final, t_final)
    # fig_graph.savefig("network_temperature_graph.png", dpi=150)
    # # fig_graph.show()  # à activer si tu veux l'afficher interactif

if __name__ == "__main__":
    run_simulation()
