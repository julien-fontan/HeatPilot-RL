import numpy as np
import os
import json
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from district_heating_gym_env import HeatNetworkEnv
import config # Import module entier pour patching
import s3fs

# Dossier local pour les modèles RL
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT_DIR = os.path.join(BASE_DIR, "models")
CHECKPOINT_PATH_S3 = "julienfntn/"  # si entrainement sur SSP Cloud

class SaveAndEvalCallback(BaseCallback):
    """
    Callback pour sauvegarder le modèle tous les N pas de temps.
    """
    def __init__(self, check_freq: int, save_dir: str, run_name: str, use_s3: bool, verbose=1):
        super(SaveAndEvalCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_dir = save_dir
        self.run_name = run_name
        self.use_s3 = use_s3
        # FS S3 initialisé seulement si nécessaire
        self.fs = s3fs.S3FileSystem(anon=False) if self.use_s3 else None

    def _init_callback(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

    def _upload_to_s3(self, local_path: str, s3_rel_path: str):
        if not self.use_s3: return
        s3_full_path = f"{CHECKPOINT_PATH_S3}/{s3_rel_path}"
        s3_url = f"s3://{s3_full_path}"
        with open(local_path, "rb") as f_local, self.fs.open(s3_url, "wb") as f_s3:
            f_s3.write(f_local.read())

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            iteration = self.num_timesteps // self.model.n_steps
            
            # Pattern: model-type_nom-personnalisé_nb-iterations
            model_basename = f"{self.run_name}_{iteration}"
            model_path_local = os.path.join(self.save_dir, model_basename)
            self.model.save(model_path_local)
            
            vec_env = self.model.get_env()
            norm_path_local = os.path.join(self.save_dir, f"vec_normalize_{self.run_name}_{iteration}.pkl")
            if isinstance(vec_env, VecNormalize):
                vec_env.save(norm_path_local)
            
            if self.verbose > 0:
                print(f"Modèle sauvegardé dans {model_basename}.zip (iter {iteration})")

            # Upload S3 (simplifié pour l'exemple, garde la structure plate sur S3 ou adapte selon besoin)
            # Ici on met tout dans le même bucket S3, on pourrait créer des sous-dossiers S3 aussi.
            model_zip_local = model_path_local + ".zip"
            self._upload_to_s3(model_zip_local, f"{model_basename}.zip")
            if os.path.exists(norm_path_local):
                self._upload_to_s3(norm_path_local, f"vec_normalize_{self.run_name}_{iteration}.pkl")

        return True

def _find_latest_model_in_dir(directory: str, run_name: str) -> str | None:
    """Cherche le dernier modèle dans un sous-dossier spécifique."""
    if not os.path.isdir(directory): return None
    candidates = []
    # On cherche les fichiers qui commencent par le run_name
    prefix = f"{run_name}_"
    for fname in os.listdir(directory):
        if fname.startswith(prefix) and fname.endswith(".zip"):
            try:
                # Extraction de l'itération à la fin
                # ex: PPO_test_1800.zip -> 1800
                suffix = fname[len(prefix):-4]
                iteration = int(suffix)
                candidates.append((iteration, fname))
            except ValueError:
                continue
    if not candidates: return None
    best_iter, best_fname = max(candidates, key=lambda x: x[0])
    return os.path.join(directory, best_fname[:-4])

def save_run_config(directory: str, config_module):
    """Sauvegarde les paramètres de config.py dans un JSON."""
    data = {
        "GLOBAL_SEED": config_module.GLOBAL_SEED,
        "EDGES": config_module.EDGES,
        # EDGES est une liste de tuples, JSON le transformera en liste de listes.
        # Il faudra le reconvertir au chargement.
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
    print(f"Configuration sauvegardée dans {path}")

def load_and_patch_config(directory: str):
    """Charge le JSON et écrase les variables du module config."""
    path = os.path.join(directory, "run_config.json")
    if not os.path.exists(path):
        print("ATTENTION: Pas de fichier run_config.json trouvé. Utilisation de config.py actuel.")
        return

    with open(path, 'r') as f:
        data = json.load(f)
    
    print(f"Chargement de la configuration depuis {path}...")
    
    # Patching du module config
    config.GLOBAL_SEED = data["GLOBAL_SEED"]
    config.PHYSICAL_PROPS = data["PHYSICAL_PROPS"]
    config.SIMULATION_PARAMS = data["SIMULATION_PARAMS"]
    config.POWER_PROFILE_CONFIG = data["POWER_PROFILE_CONFIG"]
    config.CONTROL_PARAMS = data["CONTROL_PARAMS"]
    config.TRAINING_PARAMS = data["TRAINING_PARAMS"]
    config.REWARD_PARAMS = data["REWARD_PARAMS"]
    
    # Reconversion de EDGES (liste de listes -> liste de tuples)
    edges_list = data["EDGES"]
    config.EDGES = [tuple(e) for e in edges_list]
    
    print("Configuration appliquée.")

def get_user_selection():
    """Menu interactif pour choisir ou créer un modèle."""
    os.makedirs(MODELS_ROOT_DIR, exist_ok=True)
    
    # Lister les sous-dossiers
    subdirs = [d for d in os.listdir(MODELS_ROOT_DIR) if os.path.isdir(os.path.join(MODELS_ROOT_DIR, d))]
    subdirs.sort()
    
    print("\n--- SÉLECTION DU MODÈLE ---")
    print("0. Créer un nouveau modèle")
    for i, d in enumerate(subdirs):
        print(f"{i+1}. {d}")
    
    choice = input("\nVotre choix (numéro) : ")
    try:
        choice_idx = int(choice)
    except ValueError:
        choice_idx = -1
        
    if choice_idx == 0 or not subdirs:
        # Création
        custom_name = input("Entrez un nom personnalisé : ").strip()
        if not custom_name: custom_name = "default"
        model_type = "PPO" # Pour l'instant on utilise PPO
        run_name = f"{model_type}_{custom_name}"
        run_dir = os.path.join(MODELS_ROOT_DIR, run_name)
        
        if os.path.exists(run_dir):
            print(f"Le dossier {run_name} existe déjà. Reprise de cet entraînement.")
            return run_dir, run_name, True # True = resume
        else:
            os.makedirs(run_dir)
            return run_dir, run_name, False # False = new
            
    elif 1 <= choice_idx <= len(subdirs):
        # Reprise
        run_name = subdirs[choice_idx-1]
        run_dir = os.path.join(MODELS_ROOT_DIR, run_name)
        return run_dir, run_name, True
    else:
        print("Choix invalide. Création par défaut.")
        return os.path.join(MODELS_ROOT_DIR, "PPO_default"), "PPO_default", False

def train():
    # 1. Menu interactif
    run_dir, run_name, is_resume = get_user_selection()
    
    # 2. Gestion de la config
    if is_resume:
        load_and_patch_config(run_dir)
    else:
        save_run_config(run_dir, config)

    # Paramètres après patching éventuel
    episode_length = config.TRAINING_PARAMS["episode_length_steps"]
    n_steps = config.TRAINING_PARAMS["n_steps_update"]
    total_timesteps = config.TRAINING_PARAMS["total_timesteps"]
    use_s3 = config.TRAINING_PARAMS.get("use_s3_checkpoints", False)
    normalize_env = config.TRAINING_PARAMS.get("normalize_env", True)
    
    np.random.seed(config.GLOBAL_SEED)
    
    # Création de l'environnement (utilisera la config patchée)
    env = HeatNetworkEnv()
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    if normalize_env:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.99)
        print("Environnement normalisé (VecNormalize).")

    # Callback
    save_freq_episodes = config.TRAINING_PARAMS["save_freq_episodes"]
    save_freq_steps = save_freq_episodes * episode_length
    
    callback = SaveAndEvalCallback(
        check_freq=save_freq_steps,
        save_dir=run_dir,
        run_name=run_name,
        use_s3=use_s3,
    )

    # 3. Chargement ou Création du modèle
    latest_model = _find_latest_model_in_dir(run_dir, run_name)
    
    reset_num_timesteps = True
    
    if latest_model is not None:
        print(f"Reprise de l'entraînement depuis {os.path.basename(latest_model)}")
        model = PPO.load(latest_model, env=env)
        reset_num_timesteps = False
        
        if normalize_env:
            # Chargement stats VecNormalize
            # Pattern: vec_normalize_RUNNAME_ITER.pkl
            iter_str = latest_model.split("_")[-1]
            norm_path = os.path.join(run_dir, f"vec_normalize_{run_name}_{iter_str}.pkl")
            if os.path.exists(norm_path):
                print(f"Chargement stats normalisation: {os.path.basename(norm_path)}")
                env = VecNormalize.load(norm_path, env.venv)
                model.set_env(env)
            else:
                print("Attention: Stats normalisation introuvables.")
        
        # Force params
        model.n_steps = n_steps
        model.learning_rate = config.TRAINING_PARAMS["learning_rate"]
        start_timesteps = model.num_timesteps
    else:
        print(f"Création d'un nouveau modèle PPO dans {run_dir}")
        # Récupération du coeff d'entropie depuis la config, défaut à 0.01 si absent
        ent_coef = config.TRAINING_PARAMS.get("ent_coef", 0.01)
        
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=n_steps,
            batch_size=64,
            verbose=1,
            learning_rate=config.TRAINING_PARAMS["learning_rate"],
            seed=config.GLOBAL_SEED,
            gamma=0.995,
            gae_lambda=0.98,
            ent_coef=ent_coef, # Utilisation du paramètre de config
        )
        start_timesteps = 0

    print(f"Début entraînement... (déjà fait: {start_timesteps})")
    model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=reset_num_timesteps)
    print("Entraînement terminé.")

    # Sauvegarde finale
    final_iteration = model.num_timesteps // n_steps
    final_basename = f"{run_name}_{final_iteration}"
    final_path = os.path.join(run_dir, final_basename)
    model.save(final_path)
    
    final_norm = os.path.join(run_dir, f"vec_normalize_{run_name}_{final_iteration}.pkl")
    env.save(final_norm)
    print(f"Sauvegarde finale: {final_basename}")

if __name__ == "__main__":
    train()
