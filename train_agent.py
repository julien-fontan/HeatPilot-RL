import numpy as np
import os
import json
import warnings
import s3fs

# Stable Baselines 3
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Imports du projet
from district_heating_gym_env import HeatNetworkEnv
import config

# Dossier local pour les modèles RL
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT_DIR = os.path.join(BASE_DIR, "models")
CHECKPOINT_PATH_S3 = "julienfntn/"

class SaveAndEvalCallback(BaseCallback):
    """
    Callback pour sauvegarder le modèle et les stats de normalisation tous les N pas.
    Gère aussi l'upload S3 si activé.
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
            if self.verbose > 0:
                print(f"Erreur Upload S3: {e}")

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            iteration = self.num_timesteps
            
            # Sauvegarde Modèle
            model_basename = f"{self.run_name}_{iteration}"
            model_path_local = os.path.join(self.save_dir, model_basename)
            self.model.save(model_path_local)
            
            # Sauvegarde VecNormalize (TOUJOURS actif)
            norm_path_local = os.path.join(self.save_dir, f"vec_normalize_{self.run_name}_{iteration}.pkl")
            vec_env = self.model.get_vec_normalize_env()
            if vec_env is not None:
                vec_env.save(norm_path_local)
            else:
                warnings.warn("Attention: Impossible de récupérer VecNormalize pour la sauvegarde.")
            
            if self.verbose > 0:
                print(f"Checkpoint sauvegardé: {model_basename}")

            # Upload S3
            self._upload_to_s3(model_path_local + ".zip", f"{model_basename}.zip")
            if os.path.exists(norm_path_local):
                self._upload_to_s3(norm_path_local, f"vec_normalize_{self.run_name}_{iteration}.pkl")

        return True

def _find_latest_model_in_dir(directory: str, run_name: str) -> str | None:
    """Cherche le checkpoint le plus récent."""
    if not os.path.isdir(directory): return None
    candidates = []
    prefix = f"{run_name}_"
    for fname in os.listdir(directory):
        if fname.startswith(prefix) and fname.endswith(".zip"):
            try:
                suffix = fname[len(prefix):-4]
                iteration = int(suffix)
                candidates.append((iteration, fname))
            except ValueError:
                continue
    if not candidates: return None
    best_iter, best_fname = max(candidates, key=lambda x: x[0])
    return os.path.join(directory, best_fname[:-4])

def save_run_config(directory: str, config_module):
    """Snapshot de la configuration pour reproductibilité."""
    data = {
        "GLOBAL_SEED": config_module.GLOBAL_SEED,
        "EDGES": config_module.EDGES,
        "PHYSICAL_PROPS": config_module.PHYSICAL_PROPS,
        "SIMULATION_PARAMS": config_module.SIMULATION_PARAMS,
        "POWER_PROFILE_CONFIG": config_module.POWER_PROFILE_CONFIG,
        "CONTROL_PARAMS": config_module.CONTROL_PARAMS,
        "TRAINING_PARAMS": config_module.TRAINING_PARAMS,
        "REWARD_PARAMS": config_module.REWARD_PARAMS,
    }
    path = os.path.join(directory, "run_config.json")
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Config sauvegardée: {path}")

def load_and_patch_config(directory: str):
    """Charge la config JSON et écrase les variables du module `config`."""
    path = os.path.join(directory, "run_config.json")
    if not os.path.exists(path):
        warnings.warn("Pas de run_config.json trouvé. Utilisation de config.py par défaut.")
        return

    with open(path, 'r') as f:
        data = json.load(f)
    
    print(f"Patching de la configuration depuis {path}...")
    
    config.GLOBAL_SEED = data["GLOBAL_SEED"]
    config.PHYSICAL_PROPS = data["PHYSICAL_PROPS"]
    config.SIMULATION_PARAMS = data["SIMULATION_PARAMS"]
    config.POWER_PROFILE_CONFIG = data["POWER_PROFILE_CONFIG"]
    config.CONTROL_PARAMS = data["CONTROL_PARAMS"]
    config.TRAINING_PARAMS = data["TRAINING_PARAMS"]
    config.REWARD_PARAMS = data["REWARD_PARAMS"]
    
    edges_list = data["EDGES"]
    config.EDGES = [tuple(e) for e in edges_list]
    
def get_user_selection():
    """Interface CLI simple."""
    os.makedirs(MODELS_ROOT_DIR, exist_ok=True)
    subdirs = sorted([d for d in os.listdir(MODELS_ROOT_DIR) if os.path.isdir(os.path.join(MODELS_ROOT_DIR, d))])
    
    print("\n--- SÉLECTION DU MODÈLE ---")
    print("0. [NOUVEAU] Créer un nouveau modèle")
    for i, d in enumerate(subdirs):
        print(f"{i+1}. {d}")
    
    choice = input("\nChoix : ")
    try:
        idx = int(choice)
    except ValueError:
        idx = -1
        
    if idx == 0 or not subdirs:
        custom_name = input("Nom du run (ex: 'test_v1') : ").strip() or "default"
        run_name = f"PPO_{custom_name}"
        run_dir = os.path.join(MODELS_ROOT_DIR, run_name)
        
        if os.path.exists(run_dir):
            print(f"Le dossier {run_name} existe. Mode reprise.")
            return run_dir, run_name, True
        else:
            os.makedirs(run_dir)
            return run_dir, run_name, False
            
    elif 1 <= idx <= len(subdirs):
        run_name = subdirs[idx-1]
        run_dir = os.path.join(MODELS_ROOT_DIR, run_name)
        return run_dir, run_name, True
    else:
        return os.path.join(MODELS_ROOT_DIR, "PPO_default"), "PPO_default", False

def train():
    # 1. Setup
    run_dir, run_name, is_resume = get_user_selection()
    
    if is_resume:
        load_and_patch_config(run_dir)
    else:
        save_run_config(run_dir, config)

    # Récupération params
    n_steps = config.TRAINING_PARAMS["n_steps_update"]
    total_timesteps = config.TRAINING_PARAMS["total_timesteps"]
    
    np.random.seed(config.GLOBAL_SEED)
    
    # 2. Création et validation de l'Environnement
    raw_env = HeatNetworkEnv()
    
    try:
        check_env(raw_env)
    except Exception as e:
        print(f"ERREUR check_env(): {e}")
        return

    # Wrappers
    env = Monitor(raw_env, filename=None)   # Monitor calcule les stats pour l'affichage terminal
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.99)

    # 3. Callback
    episode_length = config.TRAINING_PARAMS["episode_length_steps"]
    save_freq_steps = config.TRAINING_PARAMS["save_freq_episodes"] * episode_length
    
    callback = SaveAndEvalCallback(
        check_freq=save_freq_steps,
        save_dir=run_dir,
        run_name=run_name,
        use_s3=config.TRAINING_PARAMS.get("use_s3_checkpoints", False),
    )

    # 4. Chargement / Init modèle
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
            warnings.warn(f"ATTENTION CRITIQUE: Fichier de normalisation introuvable ({norm_path}). "
                          "Les statistiques de normalisation repartent de zéro.")
            
        model.n_steps = n_steps
        model.learning_rate = config.TRAINING_PARAMS["learning_rate"]
        model.ent_coef = config.TRAINING_PARAMS.get("ent_coef", 0.0)
        
    else:
        print(f"Initialisation nouveau modèle PPO ({run_name}) avec normalisation")
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=n_steps,
            batch_size=64,
            verbose=1, 
            learning_rate=config.TRAINING_PARAMS["learning_rate"],
            seed=config.GLOBAL_SEED,
            gamma=config.TRAINING_PARAMS["gamma"],
            gae_lambda=0.95,
            ent_coef=config.TRAINING_PARAMS.get("ent_coef", 0.01),
            device="auto"
        )
    new_logger = configure(run_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    # 5. Entraînement
    print(f"Démarrage Learn (Total Timesteps: {total_timesteps})...")
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=reset_timesteps)
        print("Entraînement terminé avec succès.")
    except KeyboardInterrupt:
        print("Entraînement interrompu par l'utilisateur.")

    # 6. Sauvegarde finale
    final_step = model.num_timesteps
    final_name = f"{run_name}_{final_step}"
    model.save(os.path.join(run_dir, final_name))
    
    norm_path_final = os.path.join(run_dir, f"vec_normalize_{run_name}_{final_step}.pkl")
    env.save(norm_path_final)
    
    print("Sauvegarde finale effectuée (Modèle + Normalisation).")

if __name__ == "__main__":
    train()