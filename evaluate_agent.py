import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from district_heating_gym_env import HeatNetworkEnv
import config

# --- CONFIGURATION ---
MODEL_SUBDIR = "PPO_test6"
MODEL_ITER = 1900800
# ---------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def find_model_path(subdir, iteration=None):
    run_dir = os.path.join(MODELS_ROOT_DIR, subdir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Dossier introuvable : {run_dir}")

    if iteration is not None:
        model_name = f"{subdir}_{iteration}"
        model_path = os.path.join(run_dir, model_name)
        if not os.path.exists(model_path + ".zip"):
             raise FileNotFoundError(f"Modèle introuvable : {model_path}.zip")
        return model_path, iteration

    candidates = []
    prefix = f"{subdir}_"
    for fname in os.listdir(run_dir):
        if fname.startswith(prefix) and fname.endswith(".zip"):
            try:
                suffix = fname[len(prefix):-4]
                candidates.append(int(suffix))
            except ValueError: continue
    
    if not candidates:
        raise FileNotFoundError(f"Aucun modèle dans {run_dir}")
    
    best_iter = max(candidates)
    return os.path.join(run_dir, f"{subdir}_{best_iter}"), best_iter

def main():
    if not MODEL_SUBDIR:
        print("Erreur: MODEL_SUBDIR requis.")
        return

    try:
        model_path, iteration = find_model_path(MODEL_SUBDIR, MODEL_ITER)
        print(f"Modèle : {model_path}.zip (iter {iteration})")
    except FileNotFoundError as e:
        print(e)
        return

    run_dir = os.path.dirname(model_path)
    cache_file = os.path.join(run_dir, f"eval_data_{MODEL_SUBDIR}_{iteration}.npz")
    
    history = {}
    
    if os.path.exists(cache_file):
        print(f"Chargement cache : {os.path.basename(cache_file)}")
        data = np.load(cache_file, allow_pickle=True)
        for k in data.files: history[k] = data[k]
    else:
        print("Lancement évaluation...")
        
        # Création Env
        env = HeatNetworkEnv()
        env = DummyVecEnv([lambda: env])
        
        # Chargement Normalisation
        norm_path = os.path.join(run_dir, f"vec_normalize_{MODEL_SUBDIR}_{iteration}.pkl")
        if os.path.exists(norm_path):
            print(f"Normalisation chargée : {os.path.basename(norm_path)}")
            env = VecNormalize.load(norm_path, env)
            env.training = False
            env.norm_reward = False
        
        model = PPO.load(model_path)
        
        obs = env.reset()
        
        # Buffers
        t_list, T_in_list, m_flow_list, rew_list = [], [], [], []
        p_dem_list, p_sup_list = [], []
        cons_temps_list = []
        
        real_env = env.envs[0].unwrapped
        
        print("Simulation épisode...")
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_array, _ = env.step(action)
            
            real_env = env.envs[0].unwrapped
            
            t_list.append(real_env.current_t)
            T_in_list.append(real_env.actual_inlet_temp)
            m_flow_list.append(real_env.actual_mass_flow)
            rew_list.append(reward[0])
            p_dem_list.append(real_env.last_total_p_demand)
            p_sup_list.append(real_env.last_total_p_supplied)
            
            # --- Extraction Températures Consommateurs ---
            # Avec le moteur vectorisé, on doit récupérer la dernière cellule du bon pipe
            temps = []
            parents_map = real_env.graph.get_parent_nodes()
            # On recalcule les températures nodales actuelles pour être précis
            # (step() les a calculées mais on ne les a pas directement dans l'attribut public simple)
            # Option plus simple: utiliser les indices du graph comme dans _get_obs
            
            # Pour l'évaluation graphique, on veut la température T_out du fluide qui arrive au consommateur
            # C'est la dernière cellule du pipe d'entrée.
            for node_id in real_env.consumer_nodes:
                parents = parents_map.get(node_id, [])
                if parents:
                    p_node = parents[0]
                    edge = real_env.graph.edges[(p_node, node_id)]
                    pipe_idx = edge["pipe_index"]
                    # Slice du pipe
                    sl = real_env.network.pipe_slices[pipe_idx]
                    # Dernière cellule
                    idx_last_cell = sl.stop - 1
                    T_val = real_env.state[idx_last_cell]
                    temps.append(T_val)
                else:
                    temps.append(20.0)
            
            cons_temps_list.append(temps)

            if done_array[0]: break
        
        history["time"] = np.array(t_list)[:-1]
        history["T_inlet"] = np.array(T_in_list)[:-1]
        history["mass_flow"] = np.array(m_flow_list)[:-1]
        history["rewards"] = np.array(rew_list)[:-1]
        history["consumer_temps"] = np.array(cons_temps_list)[:-1]
        history["p_demand"] = np.array(p_dem_list)[:-1]
        history["p_supplied"] = np.array(p_sup_list)[:-1]
        
        np.savez(cache_file, **history)
        print("Données sauvegardées.")

    # --- Plotting ---
    t_h = history["time"] / 3600.0
    
    plt.figure(figsize=(10, 10))
    
    plt.subplot(4, 1, 1)
    plt.plot(t_h, history["T_inlet"], label="T source")
    # Plot moyenne consommateurs
    T_cons_mean = np.mean(history["consumer_temps"], axis=1)
    plt.plot(t_h, T_cons_mean, label="T mean consumers", alpha=0.7)
    plt.ylabel("Temp (°C)")
    plt.legend()
    plt.grid(True)
    plt.title(f"Evaluation: {MODEL_SUBDIR} (iter {iteration})")

    plt.subplot(4, 1, 2)
    plt.plot(t_h, history["mass_flow"], color="orange")
    plt.ylabel("Flow (kg/s)")
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(t_h, history["p_demand"]/1e3, "k--", label="Demand")
    plt.plot(t_h, history["p_supplied"]/1e3, "g", label="Supplied")
    plt.ylabel("Power (kW)")
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(t_h, history["rewards"], color="red", alpha=0.5)
    plt.ylabel("Reward")
    plt.xlabel("Time (h)")
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, f"eval_{MODEL_SUBDIR}_{iteration}.png")
    plt.savefig(plot_path)
    print(f"Graphique : {plot_path}")
    plt.show()

if __name__ == "__main__":
    main()