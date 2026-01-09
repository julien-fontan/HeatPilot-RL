import numpy as np
import os
import csv
import argparse
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from district_heating_gym_env import HeatNetworkEnv
import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def get_all_checkpoints(subdir):
    """
    Récupère la liste triée de toutes les itérations de modèles disponibles dans un sous-dossier.
    Retourne [(iteration, exact_file_path), ...].
    """
    run_dir = os.path.join(MODELS_ROOT_DIR, subdir)
    if not os.path.isdir(run_dir):
        return []

    candidates = []
    prefix = f"{subdir}_"
    for fname in os.listdir(run_dir):
        if fname.startswith(prefix) and fname.endswith(".zip"):
             try:
                 iteration = int(fname[len(prefix):-4])
                 candidates.append((iteration, os.path.join(run_dir, fname)))
             except ValueError:
                 continue
    
    return sorted(candidates, key=lambda x: x[0])

def run_simulation_for_metrics(model_path, subdir, iteration):
    """
    Exécute une simulation sur une journée pour un modèle donné.
    Retourne un dictionnaire avec le cumul énergétique journalier.
    """
    run_dir = os.path.dirname(model_path)
    # Fichier de cache pour ne pas recalculer
    cache_file = os.path.join(run_dir, f"metrics_{subdir}_{iteration}.npz")
    
    if os.path.exists(cache_file):
        print(f"   -> [CACHE] Chargement métriques itération {iteration}")
        data = np.load(cache_file)
        return dict(data)

    print(f"   -> [SIMU] Calcul métriques itération {iteration}...")
    
    # Init Env
    env = HeatNetworkEnv(fixed_seed=config.GLOBAL_SEED)
    env.randomize_geometry = False
    env = DummyVecEnv([lambda: env])
    
    # Chargement normalisation si existante
    norm_path = os.path.join(run_dir, f"vec_normalize_{subdir}_{iteration}.pkl")
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training = False     
        env.norm_reward = False

    model = PPO.load(model_path)
    # L'environnement Gym est recréé, donc seed reset implicite, etc.
    obs = env.reset()
    real_env = env.envs[0].unwrapped
    
    total_wasted = 0.0
    total_supplied = 0.0
    total_demand = 0.0
    total_deficit = 0.0
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        if dones[0]: break
        
        info = infos[0]
        # Dans l'environnement actuel, metrics est calculé à chaque pas dans env.step
        # On peut le recalculer ou l'extraire si on avait accès, mais ici on recalcule c'est plus sûr
        # pour être cohérent avec ce que l'agent a fait.
        # Simplification : on utilise les metrics internes de l'objet network si disponibles
        # ou on recalcule comme dans evaluate_agent.
        
        metrics = real_env.network.get_instant_metrics(real_env.current_t, real_env.state)
        
        # Intégration trapèze simple ou rectangulaire (ici rectangulaire, dt constant)
        dt = real_env.dt
        
        wasted = metrics["wasted"] # Watts
        supplied = metrics["supplied"]
        demand = metrics["demand"]
        deficit = max(0, demand - supplied)
        
        total_wasted += wasted * dt
        total_supplied += supplied * dt
        total_demand += demand * dt
        total_deficit += deficit * dt

    # Conversion Joules -> MWh ou MWh ? Disons MWh pour être lisible
    to_mwh = 1.0 / (3.6e9) 
    
    results = {
        "wasted_mwh": total_wasted * to_mwh,
        "supplied_mwh": total_supplied * to_mwh,
        "demand_mwh": total_demand * to_mwh,
        "deficit_mwh": total_deficit * to_mwh,
        "iteration": iteration
    }
    
    # Sauvegarde cache
    np.savez(cache_file, **results)
    
    return results

def read_progress_csv(subdir):
    """
    Lit le fichier progress.csv pour extraire les rewards moyens.
    Retourne (timesteps, rewards).
    """
    run_dir = os.path.join(MODELS_ROOT_DIR, subdir)
    progress_file = os.path.join(run_dir, "progress.csv")
    
    if not os.path.exists(progress_file):
        print(f"[WARN] Fichier progress.csv non trouvé dans {run_dir}")
        return [], []
        
    timesteps = []
    rewards = []
    
    try:
        with open(progress_file, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = row.get('time/total_timesteps')
                rew = row.get('rollout/ep_rew_mean')
                
                if ts and rew:
                    try:
                        t_val = float(ts)
                        r_val = float(rew)
                        timesteps.append(t_val)
                        rewards.append(r_val)
                    except ValueError:
                        continue
    except Exception as e:
         print(f"[WARN] Erreur lecture progress.csv: {e}")
         return [], []
         
    return timesteps, rewards

def plot_learning_curve(subdir, metrics_list, progress_data=None):
    """
    Trace les courbes d'évolution des métriques en fonction des itérations.
    S'inspire du style de graph_visualization.py.
    """
    iters = []
    wasted = []
    deficit = []

    if metrics_list:
        # Extraction des données
        iters = [m["iteration"] / 1000000.0 for m in metrics_list]
        wasted = [m["wasted_mwh"] for m in metrics_list]
        deficit = [m["deficit_mwh"] for m in metrics_list]
    
    # --- Style identique à graph_visualization ---
    col_wast = '#FBC02D'  # Jaune/Or
    col_def = '#D32F2F'   # Rouge
    col_rew = '#1976D2'   # Bleu

    # Initialisation figure
    has_progress = (progress_data is not None and len(progress_data[0]) > 0)
    
    if has_progress:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=(8, 3))
        ax2 = None

    # --- Graphe 1 : Métriques ---
    if iters:
        ax1.plot(iters, wasted, color=col_wast, linestyle='--', linewidth=1.5, label="Perdu", marker='x', markersize=4)
        ax1.plot(iters, deficit, color=col_def, linestyle='-.', linewidth=1.5, label="Déficit", marker='s', markersize=4)
        
        ax1.fill_between(iters, 0, deficit, color=col_def, alpha=0.05)
        ax1.fill_between(iters, 0, wasted, color=col_wast, alpha=0.05)
    
    ax1.set_title(f"3. Entraînement du modèle", fontsize=12, fontweight='bold', pad=10, loc='left')
    ax1.set_ylabel("Énergie journalière (MWh)")
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", frameon=True, fontsize=9, framealpha=0.9)
    # ax1.spines['top'].set_visible(False)
    # ax1.spines['right'].set_visible(False)
    
    if not has_progress:
        ax1.set_xlabel("Itérations d'entraînement (x10⁶)")

    # --- Graphe 2 : Reward (si dispo) ---
    if has_progress and ax2:
        p_steps, p_rewards = progress_data
        
        # Diviser par 1000 pour éviter les grands nombres
        p_rewards_scaled = [r / 1000.0 for r in p_rewards]
        p_steps_scaled = [s / 1000000.0 for s in p_steps]
        
        ax2.plot(p_steps_scaled, p_rewards_scaled, color=col_rew, linestyle='-', linewidth=1.5, label="Reward moyen")
        
        ax2.set_xlabel("Itérations d'entraînement (x10⁶)")
        ax2.set_ylabel("Reward journalier (x10³)")
        ax2.grid(True, alpha=0.3)
        # ax2.spines['top'].set_visible(False)
        # ax2.spines['right'].set_visible(False)

    # Sauvegarde
    out_file = os.path.join(PLOTS_DIR, f"summary_eval_{subdir}.svg")
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches='tight')
    print(f"\n[OK] Graphique de synthèse sauvegardé : {out_file}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Analyse l'évolution des performances d'un modèle RL.")
    parser.add_argument("--run", type=str, required=True, help="Nom du dossier modèle (ex: PPO_26)")
    args = parser.parse_args()

    subdir = args.run
    print(f"--- Analyse du modèle : {subdir} ---")
    
    checkpoints = get_all_checkpoints(subdir)
    
    if not checkpoints:
        print(f"[ERREUR] Aucun checkpoint trouvé dans models/{subdir}")
        return

    print(f"Trouvé {len(checkpoints)} checkpoints.")
    
    metrics_list = []
    
    # Pour chaque checkpoint, on lance (ou charge) la simulation
    for it, path in checkpoints:
        m = run_simulation_for_metrics(path, subdir, it)
        metrics_list.append(m)
    
    # Tri par itération pour le plot
    metrics_list.sort(key=lambda x: x["iteration"])
    
    # Lecture des rewards depuis progress.csv
    progress_data = read_progress_csv(subdir)
    
    plot_learning_curve(subdir, metrics_list, progress_data)

if __name__ == "__main__":
    main()