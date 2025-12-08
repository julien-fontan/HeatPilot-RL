import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from stable_baselines3 import PPO
from district_heating_gym_env import HeatNetworkEnv
from config import EDGES  # éventuellement inutile mais conservé si tu veux afficher les ids de noeuds

MODELS_DIR = "./models"

def _find_latest_model(models_dir: str) -> str | None:
    """
    Retourne le chemin (sans .zip) du modèle PPO_step_* avec le plus grand nombre de timesteps.
    """
    if not os.path.isdir(models_dir):
        return None
    candidates = []
    for fname in os.listdir(models_dir):
        if fname.startswith("PPO_step_") and fname.endswith(".zip"):
            try:
                step = int(fname[len("PPO_step_"):-4])
                candidates.append((step, fname))
            except ValueError:
                continue
    if not candidates:
        return None
    best_step, best_fname = max(candidates, key=lambda x: x[0])
    return os.path.join(models_dir, best_fname[:-4])

def _resolve_model_path(preferred_path: str | None) -> str | None:
    """
    Stratégie:
      1) si preferred_path fourni et existe → l'utiliser,
      2) sinon prendre le PPO_step_* de plus grand timestep dans MODELS_DIR.
    Retourne le chemin SANS extension .zip ou None si rien trouvé.
    """
    if preferred_path:
        if os.path.exists(preferred_path + ".zip"):
            return preferred_path
        if os.path.exists(preferred_path):
            # si l'utilisateur a déjà donné le .zip complet
            return os.path.splitext(preferred_path)[0]

    latest = _find_latest_model(MODELS_DIR)
    if latest is not None:
        print(f"Aucun modèle explicite trouvé, utilisation du dernier modèle: {latest}.zip")
        return latest

    return None

def evaluate_and_plot(model_path: str | None = None):
    """
    Charge un modèle, exécute un épisode complet et affiche les résultats.
    model_path: chemin SANS extension .zip (comme dans PPO.load), ou None pour "dernier modèle".
    """
    resolved = _resolve_model_path(model_path)
    if resolved is None:
        print("Erreur: aucun modèle PPO_step_*.zip trouvé dans ./models.")
        return

    if model_path is None or resolved != model_path:
        print(f"Chemin modèle résolu: {resolved}.zip")

    # 1. Charger l'environnement et le modèle
    env = HeatNetworkEnv()
    model = PPO.load(resolved)

    # 2. Exécuter une simulation complète
    obs, _ = env.reset()
    done = False
    
    history = {
        "time": [],
        "T_inlet": [],
        "mass_flow": [],
        "rewards": [],
        "consumer_temps": []  # Liste de listes
    }

    print("Début de l'évaluation...")
    while not done:
        # Prédiction déterministe (pas d'exploration aléatoire)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        n_consumers = len(env.consumer_nodes)
        
        history["time"].append(env.current_t)
        history["T_inlet"].append(env.actual_inlet_temp)
        history["mass_flow"].append(env.actual_mass_flow)
        history["rewards"].append(reward)
        history["consumer_temps"].append(obs[:n_consumers])

    # 3. Tracer les courbes
    time_axis = np.array(history["time"]) / 3600.0  # Heures
    
    plt.figure(figsize=(12, 10))
    
    # Températures
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, history["T_inlet"], 'k--', label="T Inlet (Source)", linewidth=2)
    cons_temps = np.array(history["consumer_temps"])
    for i, node_id in enumerate(env.consumer_nodes):
        plt.plot(time_axis, cons_temps[:, i], label=f"Node {node_id}")
    plt.ylabel("Température (°C)")
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.title("Évolution des Températures aux noeuds consommateurs")
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
    # Appel sans argument -> utilise automatiquement le PPO_step_* avec le plus grand timestep
    evaluate_and_plot()
