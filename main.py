import numpy as np
import matplotlib.pyplot as plt
from heat_network import Pipe, DistrictHeatingNetwork
from utils import generate_step_function

def run_simulation():
    # --- Configuration ---
    rho = 1000.0
    cp = 4182.0
    dx = 10.0
    tspan = (0.0, 20000.0)
    seed = 42

    power=2e6 # watts

    # Topologie (Liste des arêtes)
    edges = [(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(3,10),(10,11),(11,12),
             (5,13),(13,14),(6,15),(15,16),(16,17),(7,18),(18,19),(8,20),(20,21),
             (21,22),(4,23),(23,24),(9,25),(25,26)]
    n_pipes = len(edges)

    rng = np.random.default_rng(seed)
    
    # Génération aléatoire des propriétés physiques
    n_segments = rng.integers(20, 61, size=n_pipes)
    lengths = dx * n_segments
    diameters = rng.uniform(0.15, 0.35, size=n_pipes)
    heat_coeffs = rng.uniform(0.90, 1.75, size=n_pipes) # (h)

    pipes_list = []
    velocity_funcs = []

    for i, (u_node, v_node) in enumerate(edges):
        # Création de la conduite
        p = Pipe(
            id=i,
            nodes=(u_node, v_node),
            length=float(lengths[i]),
            diameter=float(diameters[i]),
            heat_loss_coeff=float(heat_coeffs[i]),
            dx=dx,
            rho=rho, # Ajouté pour pré-calcul
            cp=cp    # Ajouté pour pré-calcul
        )
        pipes_list.append(p)
        
        # Vitesse variable
        v_func = generate_step_function(tspan[1], 900.0, 0.35, 0.55, seed=i)
        velocity_funcs.append(v_func)

    # Création du réseau
    network = DistrictHeatingNetwork(pipes_list, rho, cp)

    # Conditions aux limites (Source au noeud 1)
    temp_source_func = generate_step_function(tspan[1], 900.0, 70.0, 90.0, seed=seed)
    boundary_conditions = {1: temp_source_func}

    # État initial (70°C partout)
    y0 = np.full(network.total_cells, 70.0)

    print("Début de la simulation...")
    # t_eval force le solveur à interpoler les résultats sur une grille de temps régulière (ex: toutes les 10s)
    t_eval_points = np.arange(tspan[0], tspan[1], 10.0)
    sol = network.solve(tspan, y0, velocity_funcs, boundary_conditions, 
                        method="BDF", rtol=1e-4, atol=1e-8, t_eval=t_eval_points)
    print("Simulation terminée.")

    # --- Visualisation ---
    # Récupération des résultats pour la première conduite
    pipe_idx = 0
    indices = network.pipe_slices[pipe_idx]
    # sol.y a la forme (n_etats, n_instants).
    # indices est un slice ou un tableau d'indices des cellules de sla conduite.
    # sol.y[indices, :] -> (n_cellules_dans_la_conduite, n_instants)
    # En prenant [-1, :], on sélectionne la dernière cellule :
    T_pipe_output = sol.y[indices, :][-1, :]  # forme (n_instants,)

    # plt.figure(figsize=(10, 6))
    # plt.plot(sol.t, T_pipe_output, label=f"Sortie Conduite {pipe_idx}")
    # plt.plot(sol.t, sol.y[indices,:][0,:], label=f"Sortie Conduite {pipe_idx}")
    # plt.xlabel("Temps [s]")
    # plt.ylabel("Température [°C]")
    # plt.title("Evolution de la température en sortie de la première conduite")
    
    # plt.contour(sol.y[indices, :])
    np.savetxt("pipe_0_output.csv", sol.y[indices, :], delimiter=",")
    # plt.contour(sol.y[indices,:].T,levels=20)
    # plt.legend()
    # plt.grid(True)
    # plt.show()

if __name__ == "__main__":
    run_simulation()
