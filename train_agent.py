import numpy as np
import os
import json
import warnings
import s3fs

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
CHECKPOINT_PATH_S3 = "julienfntn/"

class SaveAndEvalCallback(BaseCallback):
    """
    Sauvegarde le modèle et les stats de normalisation.
    Gère l'upload S3 si configuré.
    """
    def __init__(self, check_freq: int, save_dir: str, run_name: str, use_s3: bool, verbose=1):
        super(SaveAndEvalCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_dir = save_dir
        self.run_name = run_name
        self.use_s3 = use_s3
        self.fs = s3fs.S3FileSystem(anon=False) if self.use_s3 else None

    def _init_callback(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

    def _upload_to_s3(self, local_path: str, s3_rel_path: str):
        if not self.use_s3: return
        try:
            s3_full_path = f"{CHECKPOINT_PATH_S3}/{s3_rel_path}"
            with open(local_path, "rb") as f_local, self.fs.open(f"s3://{s3_full_path}", "wb") as f_s3:
                f_s3.write(f_local.read())
        except Exception as e:
            if self.verbose > 0: print(f"Erreur Upload S3: {e}")

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            iteration = self.num_timesteps
            
            # 1. Sauvegarde Modèle
            model_basename = f"{self.run_name}_{iteration}"
            model_path_local = os.path.join(self.save_dir, model_basename)
            self.model.save(model_path_local)
            
            # 2. Sauvegarde VecNormalize (Crucial pour PPO)
            norm_path_local = os.path.join(self.save_dir, f"vec_normalize_{self.run_name}_{iteration}.pkl")
            vec_env = self.model.get_vec_normalize_env()
            if vec_env is not None:
                vec_env.save(norm_path_local)
            
            if self.verbose > 0:
                print(f"Checkpoint sauvegardé: {model_basename}")

            # 3. Upload S3
            self._upload_to_s3(model_path_local + ".zip", f"{model_basename}.zip")
            if os.path.exists(norm_path_local):
                self._upload_to_s3(norm_path_local, f"vec_normalize_{self.run_name}_{iteration}.pkl")

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
    best_iter, best_fname = max(candidates, key=lambda x: x[0])
    return os.path.join(directory, best_fname[:-4])

def save_run_config(directory: str, config_module):
    data = {k: getattr(config_module, k) for k in dir(config_module) if not k.startswith("__") and not callable(getattr(config_module, k))}
    # Filtrage simple des objets non sérialisables si besoin
    serializable_data = {}
    for k, v in data.items():
        try:
            json.dumps(v)
            serializable_data[k] = v
        except:
            pass # On ignore les objets complexes
            
    path = os.path.join(directory, "run_config.json")
    with open(path, 'w') as f:
        json.dump(serializable_data, f, indent=4)
    print(f"Config sauvegardée: {path}")

def get_user_selection():
    os.makedirs(MODELS_ROOT_DIR, exist_ok=True)
    subdirs = sorted([d for d in os.listdir(MODELS_ROOT_DIR) if os.path.isdir(os.path.join(MODELS_ROOT_DIR, d))])
    
    print("\n--- SÉLECTION DU MODÈLE ---")
    print("0. [NOUVEAU] Créer un nouveau modèle")
    for i, d in enumerate(subdirs):
        print(f"{i+1}. {d}")
    
    try:
        choice = int(input("\nChoix : "))
    except ValueError: choice = -1
        
    if choice == 0 or not subdirs:
        custom_name = input("Nom du run (ex: 'test_v1') : ").strip() or "default"
        run_name = f"PPO_{custom_name}"
        run_dir = os.path.join(MODELS_ROOT_DIR, run_name)
        if os.path.exists(run_dir):
            return run_dir, run_name, True # Resume
        else:
            os.makedirs(run_dir)
            return run_dir, run_name, False # New
            
    elif 1 <= choice <= len(subdirs):
        run_name = subdirs[choice-1]
        return os.path.join(MODELS_ROOT_DIR, run_name), run_name, True
    else:
        return os.path.join(MODELS_ROOT_DIR, "PPO_default"), "PPO_default", False

def train():
    run_dir, run_name, is_resume = get_user_selection()
    
    if not is_resume:
        save_run_config(run_dir, config)

    np.random.seed(config.GLOBAL_SEED)
    
    # Init Environnement
    raw_env = HeatNetworkEnv()
    try:
        check_env(raw_env)
    except Exception as e:
        print(f"ERREUR check_env(): {e}")
        return

    # Wrappers
    env = Monitor(raw_env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.99)

    # Callback
    episode_length = config.TRAINING_PARAMS["episode_length_steps"]
    save_freq_steps = config.TRAINING_PARAMS["save_freq_episodes"] * episode_length
    
    callback = SaveAndEvalCallback(
        check_freq=save_freq_steps,
        save_dir=run_dir,
        run_name=run_name,
        use_s3=config.TRAINING_PARAMS.get("use_s3_checkpoints", False),
    )

    # Modèle
    latest_model = _find_latest_model_in_dir(run_dir, run_name)
    reset_timesteps = True
    
    if latest_model:
        print(f"Chargement du checkpoint : {os.path.basename(latest_model)}")
        model = PPO.load(latest_model, env=env)
        reset_timesteps = False
        
        iter_str = latest_model.split("_")[-1]
        norm_path = os.path.join(run_dir, f"vec_normalize_{run_name}_{iter_str}.pkl")
        if os.path.exists(norm_path):
            print(f"Chargement VecNormalize : {os.path.basename(norm_path)}")
            env = VecNormalize.load(norm_path, env.venv)
            model.set_env(env)
        else:
            warnings.warn("VecNormalize manquant, reprise avec stats à zéro.")
    else:
        print(f"Nouveau modèle PPO : {run_name}")
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=config.TRAINING_PARAMS["n_steps_update"],
            batch_size=config.TRAINING_PARAMS["batch_size"],
            verbose=1, 
            learning_rate=config.TRAINING_PARAMS["learning_rate"],
            seed=config.GLOBAL_SEED,
            gamma=config.TRAINING_PARAMS["gamma"],
            ent_coef=config.TRAINING_PARAMS.get("ent_coef", 0.01),
        )
    
    new_logger = configure(run_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    print(f"Démarrage Learn (Total: {config.TRAINING_PARAMS['total_timesteps']})...")
    try:
        model.learn(total_timesteps=config.TRAINING_PARAMS["total_timesteps"], callback=callback, reset_num_timesteps=reset_timesteps)
    except KeyboardInterrupt:
        print("Interruption utilisateur.")
    finally:
        # Sauvegarde finale
        final_step = model.num_timesteps
        model.save(os.path.join(run_dir, f"{run_name}_{final_step}"))
        env.save(os.path.join(run_dir, f"vec_normalize_{run_name}_{final_step}.pkl"))
        print("Sauvegarde finale effectuée.")

if __name__ == "__main__":
    train()