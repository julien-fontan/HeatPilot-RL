import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from district_heating_gym_env import HeatNetworkEnv
import config

# --- CONFIGURATION ---
MODEL_SUBDIR = "PPO_normalisé1" 
STEP_SAMPLE = 1
# ---------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots_summary")
os.makedirs(PLOTS_DIR, exist_ok=True)

def get_sorted_models(subdir):
    run_dir = os.path.join(MODELS_ROOT_DIR, subdir)
    if not os.path.isdir(run_dir): raise FileNotFoundError(f"{run_dir} introuvable.")
    models = []
    prefix = f"{subdir}_"
    for fname in os.listdir(run_dir):
        if fname.startswith(prefix) and fname.endswith(".zip"):
            try:
                iteration = int(fname[len(prefix):-4])
                models.append((iteration, os.path.join(run_dir, fname)))
            except ValueError: continue
    return sorted(models, key=lambda x: x[0])

def evaluate_single_model(model_path, iteration, run_dir):
    cache_file = os.path.join(run_dir, f"eval_metrics_{MODEL_SUBDIR}_{iteration}.npz")
    
    if os.path.exists(cache_file):
        print(f"   -> Cache: {iteration}")
        d = np.load(cache_file)
        return d["boiler"], d["demand"], d["supplied"], d["reward"]

    # Simulation
    env = HeatNetworkEnv()
    env = DummyVecEnv([lambda: env])
    
    norm_path = os.path.join(run_dir, f"vec_normalize_{MODEL_SUBDIR}_{iteration}.pkl")
    if os.path.exists(norm_path):
        env = VecNormalize.load(norm_path, env)
        env.training = False
        env.norm_reward = False
    
    model = PPO.load(model_path)
    obs = env.reset()
    
    ep_reward = 0.0
    # Energie accumulée (Joules)
    e_boiler = 0.0
    e_demand = 0.0
    e_supplied = 0.0
    
    steps = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        
        real_env = env.envs[0].unwrapped
        ep_reward += reward[0]
        
        dt = real_env.dt
        e_boiler += real_env.last_p_source * dt
        e_demand += real_env.last_total_p_demand * dt
        e_supplied += real_env.last_total_p_supplied * dt
        steps += 1
        
        if done[0]: break
            
    # Joules -> kWh
    to_kwh = 1.0 / 3.6e6
    kwh_boiler = e_boiler * to_kwh
    kwh_demand = e_demand * to_kwh
    kwh_supplied = e_supplied * to_kwh
    mean_rew = ep_reward / steps
    
    np.savez(cache_file, boiler=kwh_boiler, demand=kwh_demand, supplied=kwh_supplied, reward=mean_rew)
    return kwh_boiler, kwh_demand, kwh_supplied, mean_rew

def main():
    if not MODEL_SUBDIR: return

    all_models = get_sorted_models(MODEL_SUBDIR)
    selected = all_models[::STEP_SAMPLE]
    print(f"{len(selected)} modèles à évaluer.")

    iters, b_en, d_en, s_en, rewards = [], [], [], [], []
    run_dir = os.path.dirname(selected[0][1])

    for i, (it, path) in enumerate(selected):
        print(f"[{i+1}/{len(selected)}] Iter {it}...")
        b, d, s, r = evaluate_single_model(path, it, run_dir)
        iters.append(it)
        b_en.append(b)
        d_en.append(d)
        s_en.append(s)
        rewards.append(r)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    
    ax1.set_title("Training Analysis")
    ax1.plot(iters, d_en, 'k--', label='Demand')
    ax1.plot(iters, s_en, 'g', label='Supplied')
    ax1.plot(iters, b_en, 'r', label='Boiler Input')
    ax1.set_ylabel("Total Energy (kWh)")
    ax1.legend()
    ax1.grid(True, alpha=0.5)
    
    ax2.plot(iters, rewards, 'b')
    ax2.set_ylabel("Mean Reward")
    ax2.set_xlabel("Iterations")
    ax2.grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"summary_{MODEL_SUBDIR}.png"))
    plt.show()

if __name__ == "__main__":
    main()