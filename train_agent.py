import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from district_heating_gym_env import HeatNetworkEnv
from config import EDGES, RL_TRAINING, GLOBAL_SEED
import s3fs  # <-- ajout pour SSP Cloud / S3

# Dossier local (comme avant)
CHECKPOINT_PATH_LOCAL = "./checkpoints"
CHECKPOINT_PATH_S3 = "julienfntn/checkpoints"

class SaveAndEvalCallback(BaseCallback):
    """
    Callback pour sauvegarder le modèle et générer des données d'évaluation
    tous les N pas de temps, à la fois en local et (optionnellement) sur S3.
    """
    def __init__(self, check_freq: int, save_path: str, eval_env: gym.Env, use_s3: bool, verbose=1):
        super(SaveAndEvalCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.eval_env = eval_env
        self.use_s3 = use_s3
        # FS S3 initialisé seulement si nécessaire
        self.fs = s3fs.S3FileSystem(anon=False) if self.use_s3 else None

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

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
        # On vérifie si le nombre de pas total est un multiple de la fréquence demandée
        if self.num_timesteps % self.check_freq == 0:
            # 1. Sauvegarde du modèle (local)
            model_basename = f"model_step_{self.num_timesteps}"
            model_path_local = os.path.join(self.save_path, model_basename)
            self.model.save(model_path_local)
            if self.verbose > 0:
                print(f"Modèle sauvegardé en local dans {model_path_local}")
            # 1.b Upload du checkpoint modèle sur S3 (optionnel)
            model_zip_local = model_path_local + ".zip"
            self._upload_to_s3(
                local_path=model_zip_local,
                s3_rel_path=f"{model_basename}.zip",
            )

            # 2. Évaluation complète (1 épisode) pour stocker les données
            obs, _ = self.eval_env.reset()
            done = False
            
            history = {
                "obs": [],
                "actions": [],
                "rewards": [],
                "t_inlet": [],
                "mass_flow": []
            }
            
            while not done:
                # Prédiction déterministe pour l'évaluation
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                
                history["obs"].append(obs)
                history["actions"].append(action)
                history["rewards"].append(reward)
                history["t_inlet"].append(self.eval_env.actual_inlet_temp)
                history["mass_flow"].append(self.eval_env.actual_mass_flow)
            
            # 3. Sauvegarde des données (local)
            data_basename = f"eval_data_step_{self.num_timesteps}.npz"
            data_path_local = os.path.join(self.save_path, data_basename)
            np.savez(
                data_path_local, 
                obs=np.array(history["obs"]), 
                actions=np.array(history["actions"]), 
                rewards=np.array(history["rewards"]),
                t_inlet=np.array(history["t_inlet"]),
                mass_flow=np.array(history["mass_flow"])
            )
            if self.verbose > 0:
                print(f"Données d'évaluation sauvegardées en local dans {data_path_local}")

            # 3.b Upload des données sur S3 (optionnel)
            self._upload_to_s3(
                local_path=data_path_local,
                s3_rel_path=data_basename,
            )
                
        return True

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
    
    # Environnement d'évaluation séparé
    eval_env = HeatNetworkEnv()
    
    # Vérification de la conformité avec l'API Gym
    check_env(env)
    print("Environnement validé.")

    # Instanciation du callback (enregistre local + éventuellement S3)
    callback = SaveAndEvalCallback(
        check_freq=save_freq_steps,
        save_path=CHECKPOINT_PATH_LOCAL,
        eval_env=eval_env,
        use_s3=use_s3,
    )

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        batch_size=144,              # par ex. 144 = 432 / 3
        verbose=1,
        learning_rate=RL_TRAINING["learning_rate"],
        seed=GLOBAL_SEED,
    )
    #  passer le learning rate à 1e-4 ou 5e-5 si loss explose
    
    print("Début de l'entraînement...")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    print("Entraînement terminé.")

    # Sauvegarde du modèle final (local)
    final_model_name = "ppo_heat_network_final"
    final_model_local = os.path.join(".", final_model_name)
    model.save(final_model_local)
    print(f"Modèle final sauvegardé en local dans {final_model_local}.zip")

    # Upload sur S3 optionnel
    if use_s3:
        fs = s3fs.S3FileSystem(anon=False)
        final_model_zip_local = final_model_local + ".zip"
        final_model_s3 = f"s3://{CHECKPOINT_PATH_S3}/{final_model_name}.zip"
        with open(final_model_zip_local, "rb") as f_local, fs.open(final_model_s3, "wb") as f_s3:
            f_s3.write(f_local.read())
        print(f"Modèle final uploadé sur S3: {final_model_s3}")
    else:
        print("Upload S3 désactivé (use_s3_checkpoints=False).")

    # Test rapide (inchangé)
    obs, _ = env.reset()    # remet env. à zéro, et on fait première observation
    total_reward = 0
    for _ in range(1000):
        action, _states = model.predict(obs)    # le modèle est maintenant déterministe (ne fait plus d'explo)
        obs, reward, terminated, truncated, _info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, _ = env.reset()
            
    print(f"Récompense cumulée sur le test : {total_reward}")

if __name__ == "__main__":
    train()
