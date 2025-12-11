"""
Configuration centralisée pour le réseau de chaleur et l'entraînement RL.
Contient uniquement des paramètres considérés "fixes" pour une campagne donnée.
"""

# --- Seed globale ---
GLOBAL_SEED = 42

# --- Topologie globale ---
EDGES = [
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
    (3,10), (10,11),
    (5,30), (30,31),(31,32),
]

# --- Paramètres physiques (fluides + pertes moyennes) ---
PHYSICAL_PROPS = dict(
    rho=1000.0,
    cp=4182.0,
    thermal_conductivity=0.0,
    external_temp=10.0,
)

# --- Paramètres de génération des conduites (Pipe.generate_parameters) ---
PIPE_GENERATION = dict(
    length_min=50.0,            # m, longueur minimale d'une conduite
    length_max=300.0,
    diameter_min=0.15,          # m, diamètre minimal d'une conduite
    diameter_max=0.35,
    heat_loss_coeff_min=0.90,   # W/m²/K
    heat_loss_coeff_max=1.75,
)

# Température minimale de retour (utilisée pour le soutirage de puissance ET l'état initial)
MIN_RETURN_TEMP = 40.0      # °C

# --- Conditions initiales réseau / actionneurs ---
INITIAL_CONDITIONS = dict(
    initial_temp=MIN_RETURN_TEMP,  # °C dans toutes les cellules au reset
    initial_flow=15.0,             # kg/s débit de base à la source
)

# --- Paramètres de discrétisation / simulation "physique" ---
SIMULATION_PARAMS = dict(
    dx=10,                  # m, influence énormément la durée de la simulation
    t_max_day=24 * 3600.0,  # s (un épisode simulé = 1 journée)
    rtol=1e-4,
    atol=1e-4,
    warmup=3600*3           # Durée de pré-chauffe (warmup) avant la simulation principale (s)
)

# --- Pas de contrôle RL (référence temporelle pour les rampes et les steps) ---
dt_rl = 10  # s, pas de contrôle explicite pour le RL et pour les profils "discrets"

# --- Paramètres des profils de puissance demandée par les consommateurs ---
POWER_PROFILE_CONFIG = dict(
    p_min=100_000.0,        # W
    p_max=300_000.0,        # W
    step_time=3600.0,       # s entre deux changements de demande
    smooth_factor=1.0,      # >1 => profils plus lisses
)

# nombre de pas RL entre deux changements de demande, pour information
POWER_PROFILE_CONFIG["step_time_steps"] = int(
    POWER_PROFILE_CONFIG["step_time"] / dt_rl
)

# --- Contraintes physiques / actionneurs (utilisées dans l'env Gym) ---
# On exprime les rampes en K/min et kg/s/min puis on les convertit en "par pas" via dt_rl.
_ramp_up_K_per_min = 3.0        # montée max ≈ 3 °C/min
_ramp_down_K_per_min = 3.0     # en pratique, en mélangeant le fluide avec la sortie : ≈ 20 °C/min
_ramp_flow_kgps_per_min = 3.0    # variation débit ≈ 3 kg/s par minute

CONTROL_LIMITS = dict(
    temp_min=50.0,
    temp_max=120.0,
    flow_min=3.0,
    flow_max=30.0,
    max_temp_rise_per_dt=_ramp_up_K_per_min * dt_rl / 60.0,
    max_temp_drop_per_dt=_ramp_down_K_per_min * dt_rl / 60.0,
    max_flow_delta_per_dt=_ramp_flow_kgps_per_min * dt_rl / 60.0,
)

# --- Paramètres d'entraînement RL ---
_episode_length_steps = int(SIMULATION_PARAMS["t_max_day"] / dt_rl)
_total_episodes = 200 # nombre total d'épisodes d'entraînement
_total_timesteps = _episode_length_steps * _total_episodes

# _n_steps_update : int(1*3600 / dt_rl)  # 1 heure de simulation par update
# AUGMENTATION : 1h (360 steps) est souvent trop court pour PPO. 
# On passe à 2048 steps (standard SB3) ou environ 6h de simulation pour stabiliser le gradient.
_n_steps_update = 2048 

RL_TRAINING = dict(
    dt=dt_rl,                            # pas de contrôle explicite RL
    total_timesteps=_total_timesteps,
    episode_length_steps=_episode_length_steps,
    n_steps_update=_n_steps_update,
    learning_rate=3e-4,
    ent_coef=0.05,             # AJOUT : Coefficient d'entropie (0.01 -> 0.05 pour forcer l'exploration)
    save_freq_episodes=5,      # Fréquence de sauvegarde en nombre d'épisodes
    use_s3_checkpoints=False,  # False = uniquement local, True = local + S3
    normalize_env=False,        # MODIFICATION : True aide énormément la convergence et l'exploration
)

# --- Poids de la fonction de récompense (pénalités RL) ---
# Configuration alignée avec reward_plot.py
REWARD_CONFIG = dict(
    weights=dict(
        comfort=1.0,    # Coeff A (Linéaire)
        boiler=1.0,     # Coeff B (Sobriété Boiler)
        pump=0.5        # Coeff C (Sobriété Pompage)
    ),
    params=dict(
        p_ref=2000.0,         # Puissance de référence (kW)
        p_pump_nominal=15.0   # Puissance nominale pompe (kW)
    )
)
