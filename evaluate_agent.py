import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from district_heating_gym_env import HeatNetworkEnv
from config import EDGES  # éventuellement inutile mais conservé si tu veux afficher les ids de noeuds

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def _find_latest_model(models_dir: str) -> str | None:
    """
    Retourne le chemin (sans .zip) du modèle PPO_iter_* avec le plus grand nombre d'itérations.
    """
    if not os.path.isdir(models_dir):
        return None
    candidates = []
    for fname in os.listdir(models_dir):
        if fname.startswith("PPO_iter_") and fname.endswith(".zip"):
            try:
                iteration = int(fname[len("PPO_iter_"):-4])
                candidates.append((iteration, fname))
            except ValueError:
                continue
    if not candidates:
        return None
    best_iter, best_fname = max(candidates, key=lambda x: x[0])
    return os.path.join(models_dir, best_fname[:-4])

def _resolve_model_path(model_name: str | None) -> str | None:
    """
    Stratégie:
      - Si model_name est fourni, on suppose qu'il s'agit du nom d'un modèle situé dans MODELS_DIR
        (ex: 'PPO_iter_100') et on cherche MODELS_DIR/model_name.zip.
      - Si model_name est None, on prend le PPO_iter_* de plus grand index dans MODELS_DIR.
    Retourne le chemin SANS extension .zip ou None si rien trouvé.
    """
    if model_name:
        candidate = os.path.join(MODELS_DIR, model_name)
        if os.path.exists(candidate + ".zip"):
            return candidate
        else:
            print(f"Modèle '{model_name}.zip' introuvable dans {MODELS_DIR}")
            return None
    # Pas de nom fourni → on prend le dernier modèle
    latest = _find_latest_model(MODELS_DIR)
    if latest is not None:
        print(f"Aucun modèle explicite fourni, utilisation du dernier modèle: {os.path.basename(latest)}.zip")
        return latest

    return None

def evaluate_and_plot(model_path: str | None = None):
    """
    Charge un modèle, exécute un épisode complet et affiche les résultats.
    model_path: nom du modèle (sans .zip) dans le dossier ./models, ou None pour "dernier modèle".
    """
    resolved = _resolve_model_path(model_path)
    if resolved is None:
        print("Erreur: aucun modèle PPO_iter_*.zip trouvé dans ./models.")
        return

    if model_path is None or os.path.basename(resolved) != model_path:
        print(f"Chemin modèle résolu: {os.path.basename(resolved)}.zip")

    # 1. Charger l'environnement et le modèle
    # Il faut recréer l'environnement exactement comme à l'entraînement (VecNormalize)
    env = HeatNetworkEnv()
    env = DummyVecEnv([lambda: env])
    
    # Chercher le fichier de stats correspondant
    # resolved est le chemin sans extension, ex: .../PPO_iter_10
    iter_str = os.path.basename(resolved).split("_")[-1]
    norm_path = os.path.join(MODELS_DIR, f"vec_normalize_iter_{iter_str}.pkl")
    
    if os.path.exists(norm_path):
        print(f"Chargement des statistiques de normalisation: {os.path.basename(norm_path)}")
        env = VecNormalize.load(norm_path, env)
        # IMPORTANT: En évaluation, on ne met pas à jour les stats (training=False)
        # et on ne normalise pas la reward (norm_reward=False) pour lire la vraie reward physique
        env.training = False
        env.norm_reward = False
    else:
        print("INFO: Pas de fichier vec_normalize_*.pkl trouvé. Utilisation de l'environnement brut (sans normalisation).")
        # On ne wrap PAS dans VecNormalize ici, on suppose que le modèle a été entraîné sans.

    model = PPO.load(resolved)

    # 2. Exécuter une simulation complète
    obs = env.reset() # VecEnv retourne directement l'obs
    done = False
    
    history = {
        "time": [],
        "T_inlet": [],
        "mass_flow": [],
        "rewards": [],
        "consumer_temps": [],   # Liste de listes
        "p_demand_tot": [],     # W
        "p_supplied_tot": [],   # W
    }

    print("Début de l'évaluation...")
    # Attention: avec VecEnv, done est un tableau de booléens
    while True:
        # Prédiction déterministe (pas d'exploration aléatoire)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_array, _infos = env.step(action)
        
        # VecEnv reset automatiquement quand done=True. 
        # Pour l'évaluation d'un seul épisode, on check done_array[0]
        # Mais on veut récupérer les infos avant le reset automatique (stockées dans _infos[0]['terminal_observation'] si besoin)
        # Ici HeatNetworkEnv ne renvoie pas d'info critique dans info, on utilise l'env interne pour les logs
        
        # Accès à l'environnement interne (unwrapped) pour lire les vraies valeurs physiques
        real_env = env.envs[0].unwrapped
        
        n_consumers = len(real_env.consumer_nodes)
        
        history["time"].append(real_env.current_t)
        history["T_inlet"].append(real_env.actual_inlet_temp)
        history["mass_flow"].append(real_env.actual_mass_flow)
        history["rewards"].append(reward[0]) # reward est un array
        
        # obs est normalisé, donc difficile à lire directement. 
        # On préfère lire l'état interne ou dénormaliser si besoin.
        # Ici on lit real_env.state pour les températures consommateurs
        # Attention: real_env.state contient TOUTES les cellules. Il faut extraire les noeuds.
        # Plus simple : utiliser real_env._get_obs() mais il renvoie aussi du normalisé/clippé partiel.
        # On va reconstruire les températures consommateurs depuis real_env
        
        cons_temps = []
        parents_map = real_env.graph.get_parent_nodes()
        for node_id in real_env.consumer_nodes:
            parents = parents_map.get(node_id, [])
            if parents:
                parent_node = parents[0]
                edge_data = real_env.graph.edges[(parent_node, node_id)]
                pipe_idx = edge_data["pipe_index"]
                sl = real_env.network.pipe_slices[pipe_idx]
                cons_temps.append(real_env.state[sl.stop - 1])
            else:
                cons_temps.append(20.0)
        history["consumer_temps"].append(cons_temps)

        history["p_demand_tot"].append(real_env.last_total_p_demand)
        history["p_supplied_tot"].append(real_env.last_total_p_supplied)

        if done_array[0]:
            break

    print(history["T_inlet"])
    print(history["mass_flow"])

    # 3. Tracer les courbes principales (températures, débit, reward)
    time_axis = np.array(history["time"]) / 3600.0  # Heures
    
    plt.figure(figsize=(12, 10))
    
    # Températures
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, history["T_inlet"], 'k--', label="T Inlet (Source)", linewidth=2)
    cons_temps = np.array(history["consumer_temps"])
    
    # Correction: accès à consumer_nodes via get_attr car env est un VecEnv
    consumer_nodes = env.get_attr("consumer_nodes")[0]
    for i, node_id in enumerate(consumer_nodes):
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
    output_file = os.path.join(PLOTS_DIR, "evaluation_results.png")
    plt.savefig(output_file)
    print(f"Graphiques sauvegardés dans {output_file}")
    plt.show()

    # 4. Figure dédiée aux puissances demandée / fournie
    p_demand = np.array(history["p_demand_tot"]) / 1e3     # kW
    p_supplied = np.array(history["p_supplied_tot"]) / 1e3 # kW

    plt.figure(figsize=(8, 4))
    plt.plot(time_axis, p_demand, label="P_demand_tot (kW)")
    plt.plot(time_axis, p_supplied, label="P_supplied_tot (kW)")
    plt.xlabel("Temps (h)")
    plt.ylabel("Puissance (kW)")
    plt.title("Puissances totale demandée vs fournie")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_file2 = os.path.join(PLOTS_DIR, "evaluation_powers.png")
    plt.savefig(output_file2)
    print(f"Courbes de puissance sauvegardées dans {output_file2}")
    plt.show()

if __name__ == "__main__":
    # Appel sans argument -> utilise automatiquement le PPO_step_* avec le plus grand timestep
    evaluate_and_plot()
