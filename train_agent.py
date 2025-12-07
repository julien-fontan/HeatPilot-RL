import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from district_heating_gym_env import HeatNetworkEnv
from config import EDGES, RL_TRAINING, GLOBAL_SEED

class SaveAndEvalCallback(BaseCallback):
    """
    Callback pour sauvegarder le modèle et générer des données d'évaluation tous les N pas de temps.
    """
    def __init__(self, check_freq: int, save_path: str, eval_env: gym.Env, verbose=1):
        super(SaveAndEvalCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.eval_env = eval_env

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # On vérifie si le nombre de pas total est un multiple de la fréquence demandée
        if self.num_timesteps % self.check_freq == 0:
            
            # 1. Sauvegarde du modèle
            model_path = os.path.join(self.save_path, f"model_step_{self.num_timesteps}")
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Modèle sauvegardé dans {model_path}")

            # 2. Évaluation complète (1 épisode) pour stocker les données
            obs, _ = self.eval_env.reset()
            done = False
            
            # Collecte des données
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
            
            # Sauvegarde des données dans un fichier compressé numpy (.npz)
            data_path = os.path.join(self.save_path, f"eval_data_step_{self.num_timesteps}.npz")
            np.savez(
                data_path, 
                obs=np.array(history["obs"]), 
                actions=np.array(history["actions"]), 
                rewards=np.array(history["rewards"]),
                t_inlet=np.array(history["t_inlet"]),
                mass_flow=np.array(history["mass_flow"])
            )
            if self.verbose > 0:
                print(f"Données d'évaluation sauvegardées dans {data_path}")
                
        return True

def train():

    episode_length = RL_TRAINING["episode_length_steps"]
    n_steps = RL_TRAINING["n_steps_update"]
    total_timesteps = episode_length * RL_TRAINING["total_episodes"]
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

    # Instanciation du callback
    callback = SaveAndEvalCallback(
        check_freq=save_freq_steps,
        save_path="./checkpoints/",
        eval_env=eval_env
    )

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        verbose=1,
        learning_rate=RL_TRAINING["learning_rate"],
        seed=GLOBAL_SEED,
    )
    #  passer le learning rate à 1e-4 ou 5e-5 si loss explose
    
    print("Début de l'entraînement...")
    model.learn(total_timesteps=total_timesteps, callback=callback)
    print("Entraînement terminé.")

    # Sauvegarde
    model.save("ppo_heat_network_final")

    # Test rapide
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
