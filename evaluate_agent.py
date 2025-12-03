import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from heat_gym_env import HeatNetworkEnv
from main import EDGES

def evaluate_and_plot(model_path="ppo_heat_network_final"):
    """
    Charge un modèle, exécute un épisode complet et affiche les résultats.
    """
    if not os.path.exists(model_path + ".zip"):
        print(f"Erreur: Le modèle '{model_path}' est introuvable.")
        return

    # 1. Charger l'environnement et le modèle
    env = HeatNetworkEnv(edges=EDGES)
    model = PPO.load(model_path)

    # 2. Exécuter une simulation complète
    obs, _ = env.reset()
    done = False
    
    history = {
        "time": [],
        "T_inlet": [],
        "mass_flow": [],
        "rewards": [],
        "consumer_temps": [] # Liste de listes
    }

    print("Début de l'évaluation...")
    while not done:
        # Prédiction déterministe (pas d'exploration aléatoire)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Récupération des infos pour les graphiques
        # obs structure: [temps_consommateurs..., T_in, Flow, demandes...]
        n_consumers = len(env.consumer_nodes)
        
        history["time"].append(env.current_t)
        history["T_inlet"].append(env.actual_inlet_temp)
        history["mass_flow"].append(env.actual_mass_flow)
        history["rewards"].append(reward)
        history["consumer_temps"].append(obs[:n_consumers])

    # 3. Tracer les courbes
    time_axis = np.array(history["time"]) / 3600.0 # Heures
    
    plt.figure(figsize=(12, 10))
    
    # Températures
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, history["T_inlet"], 'k--', label="T Inlet (Source)", linewidth=2)
    cons_temps = np.array(history["consumer_temps"])
    # On affiche quelques consommateurs pour ne pas surcharger
    for i, node_id in enumerate(env.consumer_nodes):
        plt.plot(time_axis, cons_temps[:, i], label=f"Node {node_id}")
    plt.ylabel("Température (°C)")
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.title("Évolution des Températures")
    plt.grid(True)

    # Débit
    plt.subplot(3, 1, 2)
    plt.plot(time_axis, history["mass_flow"], 'b-')
    plt.ylabel("Débit (kg/s)")
    plt.title("Débit Massique à la Source")
    plt.grid(True)

    # Reward
    plt.subplot(3, 1, 3)
    plt.plot(time_axis, history["rewards"], 'r-', alpha=0.6)
    plt.ylabel("Reward")
    plt.xlabel("Temps (h)")
    plt.title("Récompense Instantanée")
    plt.grid(True)

    plt.tight_layout()
    output_file = "evaluation_results.png"
    plt.savefig(output_file)
    print(f"Graphiques sauvegardés dans {output_file}")
    plt.show()

if __name__ == "__main__":
    # Vous pouvez changer le chemin vers un checkpoint spécifique
    # ex: evaluate_and_plot("checkpoints/model_step_432000")
    evaluate_and_plot("ppo_heat_network_final")
