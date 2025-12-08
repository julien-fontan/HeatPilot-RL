import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from district_heating_gym_env import HeatNetworkEnv
from config import EDGES, RL_TRAINING, GLOBAL_SEED
import s3fs  # <-- ajout pour SSP Cloud / S3

# Dossier local pour les modèles RL, basé sur le répertoire de ce fichier
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CHECKPOINT_PATH_S3 = "julienfntn/"

class SaveAndEvalCallback(BaseCallback):
    """
    Callback pour sauvegarder le modèle tous les N pas de temps
    (local + éventuellement S3). Aucune évaluation n'est réalisée ici.
    """
    def __init__(self, check_freq: int, models_dir: str, use_s3: bool, verbose=1):
        super(SaveAndEvalCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.models_dir = models_dir
        self.use_s3 = use_s3
        # FS S3 initialisé seulement si nécessaire
        self.fs = s3fs.S3FileSystem(anon=False) if self.use_s3 else None

    def _init_callback(self) -> None:
        os.makedirs(self.models_dir, exist_ok=True)

    def _upload_to_s3(self, local_path: str, s3_rel_path: str):
        """
        Copie un fichier local vers S3 sous CHECKPOINT_PATH_S3/s3_rel_path.
        Ne fait rien si use_s3=False.
        """
        if not self.use_s3:
            return
        s3_full_path = f"{CHECKPOINT_PATH_S3}/{s3_rel_path}"
        s3_url = f"s3://{s3_full_path}"
        with open(local_path, "rb") as f_local, self.fs.open(s3_url, "wb") as f_s3:
            f_s3.write(f_local.read())
        if self.verbose > 0:
            print(f"Fichier uploadé sur S3: {s3_url}")

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            model_basename = f"PPO_step_{self.num_timesteps}"
            model_path_local = os.path.join(self.models_dir, model_basename)
            self.model.save(model_path_local)
            if self.verbose > 0:
                print(f"Modèle sauvegardé dans {model_basename}.zip")

            # Upload S3 optionnel
            model_zip_local = model_path_local + ".zip"
            self._upload_to_s3(
                local_path=model_zip_local,
                s3_rel_path=f"{model_basename}.zip",
            )

        return True

def _find_latest_model(models_dir: str) -> str | None:
    """
    Retourne le chemin (sans .zip) du modèle PPO_step_* avec le plus grand nombre de timesteps.
    """
    if not os.path.isdir(models_dir):
        return None
    candidates = []
    for fname in os.listdir(models_dir):
        if fname.startswith("PPO_step_") and fname.endswith(".zip"):
            try:
                step = int(fname[len("PPO_step_"):-4])
                candidates.append((step, fname))
            except ValueError:
                continue
    if not candidates:
        return None
    best_step, best_fname = max(candidates, key=lambda x: x[0])
    return os.path.join(models_dir, best_fname[:-4])  # sans .zip

def train():

    episode_length = RL_TRAINING["episode_length_steps"]
    n_steps = RL_TRAINING["n_steps_update"]
    total_timesteps = RL_TRAINING["total_timesteps"]
    use_s3 = RL_TRAINING.get("use_s3_checkpoints", False)
    np.random.seed(GLOBAL_SEED)
    
    # Configuration de la sauvegarde
    save_freq_episodes = RL_TRAINING["save_freq_episodes"]
    save_freq_steps = save_freq_episodes * episode_length
    
    # Création de l'environnement
    env = HeatNetworkEnv()
    
    # Vérification de la conformité avec l'API Gym
    check_env(env)
    print("Environnement validé.")

    # Instanciation du callback (enregistre local + éventuellement S3)
    callback = SaveAndEvalCallback(
        check_freq=save_freq_steps,
        models_dir=MODELS_DIR,
        use_s3=use_s3,
    )

    # --- Reprise éventuelle de l'entraînement depuis le dernier PPO_step_* ---
    os.makedirs(MODELS_DIR, exist_ok=True)
    latest_model = _find_latest_model(MODELS_DIR)
    if latest_model is not None:
        latest_name = os.path.basename(latest_model) + ".zip"
        print(f"Reprise de l'entraînement depuis {latest_name}")
        model = PPO.load(latest_model, env=env)
        # Optionnel: s'assurer que certains hyperparams sont ceux de RL_TRAINING
        model.n_steps = n_steps
        model.learning_rate = RL_TRAINING["learning_rate"]
        model.seed = GLOBAL_SEED
        start_timesteps = int(latest_model.split("_")[-1])
    else:
        print(f"Aucun modèle trouvé dans {MODELS_DIR}, création d'un nouveau modèle PPO.")
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=n_steps,
            batch_size=144,              # par ex. 144 = 432 / 3
            verbose=1,
            learning_rate=RL_TRAINING["learning_rate"],
            seed=GLOBAL_SEED,
        )
        start_timesteps = 0

    print(f"Début de l'entraînement... (timesteps déjà appris: {start_timesteps})")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    print("Entraînement terminé.")

    # Sauvegarde d'un dernier snapshot au nombre total de timesteps cumulés
    final_timesteps = start_timesteps + total_timesteps
    final_basename = f"PPO_step_{final_timesteps}"
    final_model_local = os.path.join(MODELS_DIR, final_basename)
    model.save(final_model_local)
    print(f"Dernier modèle sauvegardé dans {final_basename}.zip")

    # Upload sur S3 optionnel
    if use_s3:
        fs = s3fs.S3FileSystem(anon=False)
        final_model_zip_local = final_model_local + ".zip"
        final_model_s3 = f"s3://{CHECKPOINT_PATH_S3}/{final_basename}.zip"
        with open(final_model_zip_local, "rb") as f_local, fs.open(final_model_s3, "wb") as f_s3:
            f_s3.write(f_local.read())
        print(f"Dernier modèle uploadé sur S3: {final_model_s3}")
    else:
        print("Upload S3 désactivé (use_s3_checkpoints=False).")

    # Test rapide
    obs, _ = env.reset()
    total_reward = 0
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, _info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, _ = env.reset()
    print(f"Récompense cumulée sur le test : {total_reward}")

if __name__ == "__main__":
    train()
