import numpy as np
import os
import json
import warnings
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from district_heating_gym_env import HeatNetworkEnv
import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT_DIR = os.path.join(BASE_DIR, "models")

class SaveCallback(BaseCallback):
    def __init__(self, check_freq: int, save_dir: str, run_name: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_dir = save_dir
        self.run_name = run_name

    def _init_callback(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            iteration = self.num_timesteps
            model_path = os.path.join(self.save_dir, f"{self.run_name}_{iteration}")
            self.model.save(model_path)
            
            norm_path = os.path.join(self.save_dir, f"vec_normalize_{self.run_name}_{iteration}.pkl")
            vec_env = self.model.get_vec_normalize_env()
            if vec_env is not None:
                vec_env.save(norm_path)
            
            if self.verbose > 0: print(f"Checkpoint sauvegardé: {model_path}")
        return True

def _find_latest_model_in_dir(directory: str, run_name: str) -> str | None:
    if not os.path.isdir(directory): return None
    candidates = []
    prefix = f"{run_name}_"
    for fname in os.listdir(directory):
        if fname.startswith(prefix) and fname.endswith(".zip"):
            try:
                suffix = fname[len(prefix):-4]
                candidates.append((int(suffix), fname))
            except ValueError: continue
    if not candidates: return None
    return os.path.join(directory, max(candidates, key=lambda x: x[0])[1][:-4])

def save_run_config(directory: str, config_module):
    data = {k: getattr(config_module, k) for k in dir(config_module) if not k.startswith("__") and not callable(getattr(config_module, k))}
    serializable_data = {}
    for k, v in data.items():
        try:
            json.dumps(v)
            serializable_data[k] = v
        except: pass
    with open(os.path.join(directory, "run_config.json"), 'w') as f:
        json.dump(serializable_data, f, indent=4)

def train():
    os.makedirs(MODELS_ROOT_DIR, exist_ok=True)
    subdirs = sorted([d for d in os.listdir(MODELS_ROOT_DIR) if os.path.isdir(os.path.join(MODELS_ROOT_DIR, d))])
    
    print("\n--- SÉLECTION DU MODÈLE ---")
    print("0. [NOUVEAU]")
    for i, d in enumerate(subdirs): print(f"{i+1}. {d}")
    
    try: choice = int(input("\nChoix : "))
    except ValueError: choice = -1
        
    if choice == 0 or not subdirs:
        custom_name = input("Nom du run : ").strip() or "default"
        run_name = f"PPO_{custom_name}"
        run_dir = os.path.join(MODELS_ROOT_DIR, run_name)
        os.makedirs(run_dir, exist_ok=True)
        is_resume = False
    elif 1 <= choice <= len(subdirs):
        run_name = subdirs[choice-1]
        run_dir = os.path.join(MODELS_ROOT_DIR, run_name)
        is_resume = True
    else: return

    if not is_resume: save_run_config(run_dir, config)
    np.random.seed(config.GLOBAL_SEED)
    
    raw_env = HeatNetworkEnv()
    check_env(raw_env)
    env = VecNormalize(DummyVecEnv([lambda: Monitor(raw_env)]), norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.99)

    save_callback = SaveCallback(
        check_freq=config.TRAINING_PARAMS["save_freq_episodes"] * config.TRAINING_PARAMS["episode_length_steps"],
        save_dir=run_dir, run_name=run_name
    )

    latest_model = _find_latest_model_in_dir(run_dir, run_name)
    reset_timesteps = True
    
    if latest_model:
        print(f"Reprise du checkpoint : {latest_model}")
        model = PPO.load(latest_model, env=env)
        reset_timesteps = False
        norm_path = os.path.join(run_dir, f"vec_normalize_{run_name}_{latest_model.split('_')[-1]}.pkl")
        if os.path.exists(norm_path):
            env = VecNormalize.load(norm_path, env.venv)
            model.set_env(env)
    else:
        print(f"Nouveau modèle : {run_name}")
        model = PPO("MlpPolicy", env, verbose=1, seed=config.GLOBAL_SEED, 
                    learning_rate=config.TRAINING_PARAMS["learning_rate"],
                    n_steps=config.TRAINING_PARAMS["n_steps_update"],
                    batch_size=config.TRAINING_PARAMS["batch_size"],
                    ent_coef=config.TRAINING_PARAMS["ent_coef"])
    
    model.set_logger(configure(run_dir, ["stdout", "csv", "tensorboard"]))
    try:
        model.learn(total_timesteps=config.TRAINING_PARAMS["total_timesteps"], callback=save_callback, reset_num_timesteps=reset_timesteps)
    except KeyboardInterrupt:
        print("Interruption...")
    finally:
        model.save(os.path.join(run_dir, f"{run_name}_final"))
        env.save(os.path.join(run_dir, f"vec_normalize_{run_name}_final.pkl"))

if __name__ == "__main__":
    train()