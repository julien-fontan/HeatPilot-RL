import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import re
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from district_heating_gym_env import HeatNetworkEnv
import config

# --- CONFIGURATION UTILISATEUR ---
# Nom du sous-dossier dans /models (ex: "PPO_default")
MODEL_SUBDIR = "PPO_normalisé1" 
# Pas d'échantillonnage (1 sur N)
STEP_SAMPLE = 1
# ---------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots_summary")
os.makedirs(PLOTS_DIR, exist_ok=True)

def get_sorted_models(subdir):
    """
    Récupère la liste des modèles triés par itération.
    Retourne une liste de tuples (iteration, full_path).
    """
    run_dir = os.path.join(MODELS_ROOT_DIR, subdir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Le dossier {run_dir} n'existe pas.")

    models = []
    prefix = f"{subdir}_"
    
    for fname in os.listdir(run_dir):
        if fname.startswith(prefix) and fname.endswith(".zip"):
            try:
                # Extraction de l'itération : PPO_run_1000.zip -> 1000
                suffix = fname[len(prefix):-4]
                iteration = int(suffix)
                models.append((iteration, os.path.join(run_dir, fname)))
            except ValueError:
                continue
    
    # Tri par itération croissante
    models.sort(key=lambda x: x[0])
    return models

def evaluate_single_model(model_path, iteration, run_dir):
    """
    Évalue un modèle unique et retourne les métriques agrégées.
    Utilise un cache .npz pour éviter de recalculer si déjà fait.
    """
    cache_file = os.path.join(run_dir, f"eval_metrics_{MODEL_SUBDIR}_{iteration}.npz")
    
    if os.path.exists(cache_file):
        print(f"   -> Chargement cache : {iteration}")
        data = np.load(cache_file)
        return (data["total_boiler_energy"], 
                data["total_demand_energy"], 
                data["total_supplied_energy"], 
                data["mean_reward"])

    # --- Simulation ---
    # Création de l'environnement
    env = HeatNetworkEnv()
    env = DummyVecEnv([lambda: env])
    
    # Chargement VecNormalize si présent
    norm_path = os.path.join(run_dir, f"vec_normalize_{MODEL_SUBDIR}_{iteration}.pkl")
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False
    
    model = PPO.load(model_path)
    
    obs = env.reset()
    
    # Accumulateurs
    ep_reward = 0.0
    ep_boiler_energy = 0.0
    ep_demand_energy = 0.0
    ep_supplied_energy = 0.0
    
    steps = 0
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        
        real_env = env.envs[0].unwrapped
        
        # Somme des rewards
        ep_reward += reward[0]
        
        # Intégration de l'énergie (Puissance * dt)
        # Les puissances sont en Watts, dt en secondes -> Joules
        # On convertira en kWh à la fin
        dt = real_env.dt
        ep_boiler_energy += real_env.last_p_source * dt
        ep_demand_energy += real_env.last_total_p_demand * dt
        ep_supplied_energy += real_env.last_total_p_supplied * dt
        
        steps += 1
        
        if done[0]:
            break
            
    # Conversion Joules -> kWh (1 kWh = 3.6e6 J)
    to_kwh = 1.0 / 3.6e6
    total_boiler_kwh = ep_boiler_energy * to_kwh
    total_demand_kwh = ep_demand_energy * to_kwh
    total_supplied_kwh = ep_supplied_energy * to_kwh
    mean_reward = ep_reward / steps # Reward moyen par pas de temps
    
    # Sauvegarde cache
    np.savez(cache_file, 
             total_boiler_energy=total_boiler_kwh,
             total_demand_energy=total_demand_kwh,
             total_supplied_energy=total_supplied_kwh,
             mean_reward=mean_reward)
    
    return total_boiler_kwh, total_demand_kwh, total_supplied_kwh, mean_reward

def main():
    if not MODEL_SUBDIR:
        print("Erreur: MODEL_SUBDIR non spécifié.")
        return

    print(f"Analyse du dossier : {MODEL_SUBDIR}")
    all_models = get_sorted_models(MODEL_SUBDIR)
    
    if not all_models:
        print("Aucun modèle trouvé.")
        return

    # Filtrage 1 sur 3
    selected_models = all_models[::STEP_SAMPLE]
    print(f"{len(all_models)} modèles trouvés. {len(selected_models)} sélectionnés pour évaluation.")

    iterations = []
    boiler_energies = []
    demand_energies = []
    supplied_energies = []
    mean_rewards = []

    run_dir = os.path.dirname(selected_models[0][1])

    for i, (iteration, path) in enumerate(selected_models):
        print(f"[{i+1}/{len(selected_models)}] Évaluation itération {iteration}...")
        b_en, d_en, s_en, m_rew = evaluate_single_model(path, iteration, run_dir)
        
        iterations.append(iteration)
        boiler_energies.append(b_en)
        demand_energies.append(d_en)
        supplied_energies.append(s_en)
        mean_rewards.append(m_rew)

    # --- Plotting ---q
    print("Génération des graphiques de synthèse...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    
    # Plot 1 : Énergies

    fig.suptitle(f"Training impact analysis", fontsize=14)
    ax1.plot(iterations, demand_energies, 'k--', label='Demand Energy', linewidth=2)
    ax1.plot(iterations, supplied_energies, 'g-', label='Supplied Energy', alpha=0.8)
    ax1.plot(iterations, boiler_energies, 'r-', label='Boiler Energy', alpha=0.8)
    
    ax1.set_ylabel("Total Energy (kWh)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2 : Reward Moyen
    ax2.plot(iterations, mean_rewards, label='Mean reward')
    
    ax2.set_ylabel("Mean reward per step")
    ax2.set_xlabel("Policy iterations")
    # ax2.set_title("Évolution du reward")
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    plot_filename = f"summary_eval_{MODEL_SUBDIR}.svg"
    plot_path = os.path.join(PLOTS_DIR, plot_filename)
    plt.savefig(plot_path, transparent=True)
    print(f"Graphique sauvegardé : {plot_path}")
    plt.show()

if __name__ == "__main__":
    main()
