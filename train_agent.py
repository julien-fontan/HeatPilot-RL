import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
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
            # Calcul de l'itération (cycle complet : collecte n_steps + entraînement)
            # Note : 'n_updates' dans les logs correspond aux descentes de gradient (iter * n_epochs)
            iteration = self.num_timesteps // self.model.n_steps
            
            model_basename = f"PPO_iter_{iteration}"
            model_path_local = os.path.join(self.models_dir, model_basename)
            self.model.save(model_path_local)
            
            # Sauvegarde des statistiques de normalisation (VecNormalize)
            # Important pour pouvoir réutiliser le modèle en évaluation
            vec_env = self.model.get_env()
            norm_path_local = os.path.join(self.models_dir, f"vec_normalize_iter_{iteration}.pkl")
            if isinstance(vec_env, VecNormalize):
                vec_env.save(norm_path_local)
            
            if self.verbose > 0:
                print(f"Modèle sauvegardé dans {model_basename}.zip (iter {iteration})")

            # Upload S3 optionnel
            model_zip_local = model_path_local + ".zip"
            self._upload_to_s3(
                local_path=model_zip_local,
                s3_rel_path=f"{model_basename}.zip",
            )
            # Upload stats normalisation
            if os.path.exists(norm_path_local):
                self._upload_to_s3(
                    local_path=norm_path_local,
                    s3_rel_path=f"vec_normalize_iter_{iteration}.pkl",
                )

        return True

def _find_latest_model(models_dir: str) -> str | None:
    """
    Retourne le chemin (sans .zip) du modèle PPO_iter_* avec le plus grand nombre d'itérations.
    """
    if not os.path.isdir(models_dir):
        return None
    candidates = []
    for fname in os.listdir(models_dir):
        if fname.startswith("PPO_iter_") and fname.endswith(".zip"):
            try:
                iteration = int(fname[len("PPO_iter_"):-4])
                candidates.append((iteration, fname))
            except ValueError:
                continue
    if not candidates:
        return None
    best_iter, best_fname = max(candidates, key=lambda x: x[0])
    return os.path.join(models_dir, best_fname[:-4])  # sans .zip

def train():

    episode_length = RL_TRAINING["episode_length_steps"]
    n_steps = RL_TRAINING["n_steps_update"]
    total_timesteps = RL_TRAINING["total_timesteps"]
    use_s3 = RL_TRAINING.get("use_s3_checkpoints", False)
    normalize_env = RL_TRAINING.get("normalize_env", True)  # Nouvelle option
    dt_control = RL_TRAINING.get("dt", None)
    np.random.seed(GLOBAL_SEED)
    
    # Configuration de la sauvegarde
    save_freq_episodes = RL_TRAINING["save_freq_episodes"]
    save_freq_steps = save_freq_episodes * episode_length

    # Rappel explicite des principaux paramètres RL (dont dt)
    if dt_control is not None:
        print(
            f"RL config: dt={dt_control}s, episode_length_steps={episode_length}, "
            f"total_timesteps={total_timesteps}, n_steps_update={n_steps}, "
            f"save_freq_episodes={save_freq_episodes}"
        )
    else:
        print(
            f"RL config: episode_length_steps={episode_length}, "
            f"total_timesteps={total_timesteps}, n_steps_update={n_steps}, "
            f"save_freq_episodes={save_freq_episodes}"
        )
    
    # Création de l'environnement
    # On utilise DummyVecEnv + éventuellement VecNormalize pour stabiliser l'apprentissage
    env = HeatNetworkEnv()
    # check_env(env) # check_env ne supporte pas toujours bien les wrappers complexes, on suppose env ok
    
    env = Monitor(env)  # Logs pour Tensorboard / CSV
    env = DummyVecEnv([lambda: env])
    
    if normalize_env:
        # VecNormalize normalise automatiquement :
        # 1. Les observations (centre et réduit la variance) -> remplace le scaling manuel
        # 2. Les récompenses (stabilise le gradient)
        # Note : Il ne normalise PAS les actions (c'est fait manuellement dans HeatNetworkEnv)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., gamma=0.99)
        print("Environnement validé et normalisé (VecNormalize).")
    else:
        print("Environnement validé (DummyVecEnv - Pas de normalisation).")

    # Instanciation du callback (enregistre local + éventuellement S3)
    callback = SaveAndEvalCallback(
        check_freq=save_freq_steps,
        models_dir=MODELS_DIR,
        use_s3=use_s3,
    )

    # --- Reprise éventuelle de l'entraînement depuis le dernier PPO_iter_* ---
    os.makedirs(MODELS_DIR, exist_ok=True)
    latest_model = _find_latest_model(MODELS_DIR)
    
    reset_num_timesteps = True
    
    if latest_model is not None:
        latest_name = os.path.basename(latest_model) + ".zip"
        print(f"Reprise de l'entraînement depuis {latest_name}")
        
        # Chargement du modèle
        model = PPO.load(latest_model, env=env)
        reset_num_timesteps = False  # On continue le comptage des steps/itérations
        
        # Tentative de chargement des stats de normalisation associées (si activé)
        if normalize_env:
            iter_str = latest_model.split("_")[-1]
            norm_path = os.path.join(MODELS_DIR, f"vec_normalize_iter_{iter_str}.pkl")
            if os.path.exists(norm_path):
                print(f"Chargement des statistiques de normalisation depuis {os.path.basename(norm_path)}")
                env = VecNormalize.load(norm_path, env.venv) # env.venv car env est déjà un VecNormalize
                model.set_env(env)
            else:
                print("Attention: Fichier de stats VecNormalize introuvable, reprise avec stats vierges.")

        # Optionnel: s'assurer que certains hyperparams sont ceux de RL_TRAINING
        model.n_steps = n_steps
        model.learning_rate = RL_TRAINING["learning_rate"]
        model.seed = GLOBAL_SEED
        start_timesteps = model.num_timesteps
    else:
        print(f"Aucun modèle trouvé dans {MODELS_DIR}, création d'un nouveau modèle PPO.")
        # Configuration PPO adaptée aux délais thermiques
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=n_steps,            # Désormais 2048 (défini dans config)
            batch_size=64,              # Taille de batch standard (64 ou 128 avec n_steps=2048)
            verbose=1,
            learning_rate=RL_TRAINING["learning_rate"],
            seed=GLOBAL_SEED,
            gamma=0.995,                # Gamma élevé (horizon ~30-40 min) pour capturer les effets retardés
            gae_lambda=0.98,            # GAE lambda élevé pour réduire le biais dû aux délais
            ent_coef=0.01,              # Force l'exploration pour éviter l'effondrement immédiat vers "min flow"
        )
        start_timesteps = 0
        reset_num_timesteps = True

    print(f"Début de l'entraînement... (timesteps déjà appris: {start_timesteps})")
    model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=reset_num_timesteps)
    print("Entraînement terminé.")

    # Sauvegarde d'un dernier snapshot
    final_iteration = model.num_timesteps // n_steps
    final_basename = f"PPO_iter_{final_iteration}"
    final_model_local = os.path.join(MODELS_DIR, final_basename)
    model.save(final_model_local)
    
    # Sauvegarde finale des stats
    final_norm_local = os.path.join(MODELS_DIR, f"vec_normalize_iter_{final_iteration}.pkl")
    env.save(final_norm_local)
    
    print(f"Dernier modèle sauvegardé dans {final_basename}.zip")

    # Upload sur S3 optionnel
    if use_s3:
        fs = s3fs.S3FileSystem(anon=False)
        # Model
        final_model_zip_local = final_model_local + ".zip"
        final_model_s3 = f"s3://{CHECKPOINT_PATH_S3}/{final_basename}.zip"
        with open(final_model_zip_local, "rb") as f_local, fs.open(final_model_s3, "wb") as f_s3:
            f_s3.write(f_local.read())
        # Stats
        final_norm_s3 = f"s3://{CHECKPOINT_PATH_S3}/vec_normalize_iter_{final_iteration}.pkl"
        with open(final_norm_local, "rb") as f_local, fs.open(final_norm_s3, "wb") as f_s3:
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
