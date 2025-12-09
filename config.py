"""
Configuration centralisée pour le réseau de chaleur et l'entraînement RL.
Contient uniquement des paramètres considérés "fixes" pour une campagne donnée.
"""

# --- Seed globale (partagée sim / RL) ---
GLOBAL_SEED = 42

# --- Topologie globale (partagée sim / RL) ---
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
    heat_loss_coeff=1.5,    # pas utilisé directement (voir génération aléatoire)
)

# Température minimale de retour (utilisée pour le soutirage de puissance ET l'état initial)
MIN_RETURN_TEMP = 40.0      # °C

# --- Paramètres de discrétisation / simulation "physique" ---
SIMULATION_PARAMS = dict(
    dx=10,                  # m, influence énormément la durée de la simulation
    t_max_day=24 * 3600.0,  # s (un épisode simulé = 1 journée)
    rtol=1e-4,
    atol=1e-4,
    warmup=3600*3           # Durée de pré-chauffe (warmup) avant la simulation principale (s)
)

# --- Paramètres de génération des conduites (Pipe.generate_parameters) ---
PIPE_GENERATION = dict(
    length_min=50.0,        # m, longueur minimale d'une conduite
    length_max=300.0,       # m, longueur maximale d'une conduite
    diameter_min=0.15,      # m
    diameter_max=0.35,      # m
    h_min=0.90,             # W/m²/K (ou équivalent utilisé dans Pipe)
    h_max=1.75,
)

# --- Conditions initiales réseau / actionneurs ---
INITIAL_CONDITIONS = dict(
    initial_temp=MIN_RETURN_TEMP,  # °C dans toutes les cellules au reset
    initial_flow=20.0,             # kg/s débit de base à la source
)

# --- Pas de contrôle RL (référence temporelle pour les rampes et les steps) ---
dt_rl = 10  # s, pas de contrôle explicite pour le RL et pour les profils "discrets"

# --- Paramètres des profils de puissance demandée aux consommateurs ---
POWER_PROFILE_CONFIG = dict(
    p_min=100_000.0,        # W
    p_max=300_000.0,        # W
    step_time=3600.0,       # s entre deux changements (échelle "macro")
    smooth_factor=2.0,      # >1 => profils plus lisses
)

# nombre de pas RL entre deux changements de demande, pour information
POWER_PROFILE_CONFIG["step_time_steps"] = int(
    POWER_PROFILE_CONFIG["step_time"] / dt_rl
)

# --- Contraintes physiques / actionneurs (utilisées dans l'env Gym) ---
# On exprime les rampes en K/min et kg/s/min puis on les convertit en "par pas" via dt_rl.
_ramp_up_K_per_min = 3.0        # montée max ≈ 3 °C/min
_ramp_down_K_per_min = 20.0     # descente max ≈ 20 °C/min
_ramp_flow_kgps_per_min = 3.    # variation débit ≈ 3 kg/s par minute

CONTROL_LIMITS = dict(
    temp_min=50.0,
    temp_max=110.0,
    flow_min=3.0,
    flow_max=30.0,
    max_temp_rise_per_dt=_ramp_up_K_per_min * dt_rl / 60.0,
    max_temp_drop_per_dt=_ramp_down_K_per_min * dt_rl / 60.0,
    max_flow_delta_per_dt=_ramp_flow_kgps_per_min * dt_rl / 60.0,
)

# --- Paramètres d'entraînement RL ---
_episode_length_steps = int(SIMULATION_PARAMS["t_max_day"] / dt_rl)
_total_episodes = 100
_total_timesteps = _episode_length_steps * _total_episodes
_n_steps_update = int((5 * 60) / dt_rl)    # nb de pas entre 2 mises à jour PPO (10 min)
RL_TRAINING = dict(
    dt=dt_rl,                            # pas de contrôle explicite RL
    total_timesteps=_total_timesteps,
    episode_length_steps=_episode_length_steps,
    n_steps_update=_n_steps_update,
    learning_rate=3e-4,
    save_freq_episodes=10,
    use_s3_checkpoints=False,  # False = uniquement local, True = local + S3
)
