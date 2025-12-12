import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from district_heating_gym_env import HeatNetworkEnv
import config

# --- CONFIGURATION UTILISATEUR ---
# Nom du sous-dossier dans /models (ex: "PPO_default")
# Ce dossier doit exister dans c:\Users\juli1\Documents Drive\Projets\Recherche\HeatPilot-RL\models
MODEL_SUBDIR = "PPO_normalisé1" 
# Nombre d'itérations spécifique (ex: 18000), ou None pour prendre le dernier disponible
MODEL_ITER = None
# ---------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def find_model_path(subdir, iteration=None):
    """
    Trouve le chemin du modèle .zip.
    Si iteration est None, cherche le plus grand index.
    Retourne (chemin_sans_extension, iteration).
    """
    run_dir = os.path.join(MODELS_ROOT_DIR, subdir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Le dossier {run_dir} n'existe pas.")

    if iteration is not None:
        # Construction directe
        # Pattern attendu : {subdir}_{iteration}.zip
        model_name = f"{subdir}_{iteration}"
        model_path = os.path.join(run_dir, model_name)
        if not os.path.exists(model_path + ".zip"):
             raise FileNotFoundError(f"Le modèle {model_path}.zip n'existe pas.")
        return model_path, iteration

    # Recherche automatique du dernier
    candidates = []
    prefix = f"{subdir}_"
    for fname in os.listdir(run_dir):
        if fname.startswith(prefix) and fname.endswith(".zip"):
            try:
                # fname = PPO_default_1000.zip -> suffix = 1000
                suffix = fname[len(prefix):-4]
                it = int(suffix)
                candidates.append(it)
            except ValueError:
                continue
    
    if not candidates:
        raise FileNotFoundError(f"Aucun modèle trouvé dans {run_dir} avec le préfixe {prefix}")
    
    best_iter = max(candidates)
    model_name = f"{subdir}_{best_iter}"
    return os.path.join(run_dir, model_name), best_iter

def main():
    if not MODEL_SUBDIR:
        print("Erreur: Vous devez spécifier MODEL_SUBDIR dans le script.")
        return

    # 1. Résolution du modèle
    try:
        model_path, iteration = find_model_path(MODEL_SUBDIR, MODEL_ITER)
        print(f"Modèle sélectionné : {model_path}.zip (iter {iteration})")
    except FileNotFoundError as e:
        print(e)
        return

    run_dir = os.path.dirname(model_path)
    
    # 2. Gestion du cache de données
    # Le fichier cache est stocké dans le dossier du modèle pour être réutilisé
    cache_file = os.path.join(run_dir, f"eval_data_{MODEL_SUBDIR}_{iteration}.npz")
    
    history = {}
    
    if os.path.exists(cache_file):
        print(f"Chargement des données existantes depuis : {os.path.basename(cache_file)}")
        data = np.load(cache_file, allow_pickle=True)
        # Reconstitution du dictionnaire
        for k in data.files:
            history[k] = data[k]
    else:
        print("Aucun fichier de cache trouvé. Lancement de la simulation d'évaluation...")
        
        # --- Simulation ---
        env = HeatNetworkEnv()
        env = DummyVecEnv([lambda: env])
        
        # Chargement VecNormalize si présent
        norm_path = os.path.join(run_dir, f"vec_normalize_{MODEL_SUBDIR}_{iteration}.pkl")
        if os.path.exists(norm_path):
            print(f"Chargement de la normalisation : {os.path.basename(norm_path)}")
            env = VecNormalize.load(norm_path, env)
            env.training = False
            env.norm_reward = False
        else:
            print("INFO: Pas de fichier de normalisation trouvé (vec_normalize_*.pkl).")

        print(f"Chargement du modèle PPO...")
        model = PPO.load(model_path)
        
        obs = env.reset()
        done = False
        
        # Buffers
        t_list = []
        T_in_list = []
        m_flow_list = []
        rew_list = []
        cons_temps_list = []
        p_dem_list = []
        p_sup_list = []
        p_boiler_list = []
        p_pump_list = []
        
        # Récupération des IDs consommateurs pour sauvegarde
        real_env = env.envs[0].unwrapped
        consumer_node_ids = np.array(real_env.consumer_nodes)

        print("Exécution de l'épisode...")
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_array, _ = env.step(action)
            
            # Accès à l'environnement physique (unwrapped)
            real_env = env.envs[0].unwrapped
            
            t_list.append(real_env.current_t)
            T_in_list.append(real_env.actual_inlet_temp)
            m_flow_list.append(real_env.actual_mass_flow)
            rew_list.append(reward[0])
            p_dem_list.append(real_env.last_total_p_demand)
            p_sup_list.append(real_env.last_total_p_supplied)
            p_boiler_list.append(real_env.last_p_source)
            p_pump_list.append(real_env.last_p_pump)
            
            # Températures consommateurs
            temps = []
            parents_map = real_env.graph.get_parent_nodes()
            for node_id in real_env.consumer_nodes:
                parents = parents_map.get(node_id, [])
                if parents:
                    parent_node = parents[0]
                    edge_data = real_env.graph.edges[(parent_node, node_id)]
                    pipe_idx = edge_data["pipe_index"]
                    sl = real_env.network.pipe_slices[pipe_idx]
                    # Température à la fin du tuyau arrivant au consommateur
                    temps.append(real_env.state[sl.stop - 1])
                else:
                    temps.append(20.0)
            cons_temps_list.append(temps)

            if done_array[0]:
                break
        
        # Conversion et sauvegarde
        history["time"] = np.array(t_list[:-1])
        history["T_inlet"] = np.array(T_in_list[:-1])
        history["mass_flow"] = np.array(m_flow_list[:-1])
        history["rewards"] = np.array(rew_list[:-1])
        history["consumer_temps"] = np.array(cons_temps_list[:-1])
        history["p_demand_tot"] = np.array(p_dem_list[:-1])
        history["p_supplied_tot"] = np.array(p_sup_list[:-1])
        history["p_boiler"] = np.array(p_boiler_list[:-1])
        history["p_pump"] = np.array(p_pump_list[:-1])
        history["consumer_node_ids"] = consumer_node_ids
        
        print(f"Sauvegarde des résultats dans {cache_file}")
        np.savez(cache_file, **history)

    # --- Plotting ---
    print("Génération des graphiques...")
    time_h = history["time"] / 3600.0
    
    # Fonction de moyenne mobile
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w
    
    window = 6
    # Application sur les grandeurs pilotées (T_inlet et mass_flow)
    T_in_smooth = moving_average(history["T_inlet"], window)
    m_flow_smooth = moving_average(history["mass_flow"], window)
    
    # Application sur la puissance fournie et le reward
    p_sup_smooth = moving_average(history["p_supplied_tot"], window)
    rew_smooth = moving_average(history["rewards"], window)

    # Ajustement du temps pour correspondre à la taille réduite par la convolution 'valid'
    time_smooth = time_h[window-1:]
    
    # Récupération des IDs si disponibles
    if "consumer_node_ids" in history:
        node_ids = history["consumer_node_ids"]
    else:
        node_ids = range(history["consumer_temps"].shape[1])

    # 1. Vue d'ensemble
    plt.figure(figsize=(5, 9))
    
    plt.subplot(4, 1, 1)
    plt.plot(time_smooth, T_in_smooth, label="T inlet (source)", linewidth=2)
    cons_temps = history["consumer_temps"]
    # for i, nid in enumerate(node_ids):
    #     plt.plot(time_h, cons_temps[:, i], label=f"Node {nid}", alpha=0.7)
    plt.ylabel("Temperature (°C)")
    # plt.title(f"Évaluation : {MODEL_SUBDIR} (iter {iteration})")
    plt.legend(loc='upper right', fontsize='small', ncol=3)
    plt.grid(True)
    
    plt.subplot(4, 1, 2)
    plt.plot(time_smooth, m_flow_smooth, 'b-')
    plt.ylabel("Mass flow (kg/s)")
    # plt.title("Débit massique")
    plt.grid(True)
    
    plt.subplot(4, 1, 3)
    plt.plot(time_h, history["p_demand_tot"]/1000, 'k--', label="Demand")
    # Utilisation de la version lissée pour la puissance fournie
    plt.plot(time_smooth, p_sup_smooth/1000, 'g-', label="Supplied (smoothed)", alpha=0.7)
    plt.ylabel("Power (kW)")
    # plt.title("Puissance totale consommateurs")
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.subplot(4, 1, 4)
    # Utilisation de la version lissée pour le reward
    plt.plot(time_smooth, rew_smooth, 'r-', alpha=0.8)
    plt.ylabel("Reward")
    plt.xlabel("Time (h)")
    # plt.title("Récompense instantanée")
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, f"eval_{MODEL_SUBDIR}_{iteration}.svg")
    plt.savefig(plot_path, transparent=True)
    print(f"Plot sauvegardé : {plot_path}")
    plt.show()


# if __name__ == "__main__":
main()