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
MIN_RETURN_TEMP = 40.0  # °C

# --- Paramètres de discrétisation / simulation "physique" ---
SIMULATION_PARAMS = dict(
    dx=0.01,               # m
    t_max_day=24 * 3600.0, # s (un épisode simulé = 1 journée)
    dt=0.1,               # pas de contrôle (s)
    rtol=1e-4,             # tolérance relative solveur ODE (°0.0001×70 ≈ 0.007°C)
    atol=1e-4,             # tolérance absolue solveur ODE (en °C) (négligeable par rapport aux variations de contrôle (0.5°C par pas, etc.)
    warmup=50             # Durée de pré-chauffe (warmup) avant la simulation principale (s)
)

# --- Paramètres de génération des conduites (Pipe.generate_parameters) ---
PIPE_GENERATION = dict(
    n_segments_min=20,
    n_segments_max=61,
    diameter_min=0.15,
    diameter_max=0.35,
    h_min=0.90,
    h_max=1.75,
)

# --- Conditions initiales réseau / actionneurs ---
INITIAL_CONDITIONS = dict(
    initial_temp=MIN_RETURN_TEMP,  # °C dans toutes les cellules au reset
    initial_flow=20.0,             # kg/s débit de base à la source
)

# --- Paramètres des profils de puissance demandée aux consommateurs ---
POWER_PROFILE_CONFIG = dict(
    p_min=100_000.0,     # W
    p_max=300_000.0,     # W
    step_time=7200.0,    # s entre deux changements (2h)
)

# --- Contraintes physiques / actionneurs (utilisées dans l'env Gym) ---
# On exprime les rampes en K/min et kg/s/min puis on les convertit en "par pas" via dt.
_dt = SIMULATION_PARAMS["dt"]
_ramp_up_K_per_min = 3.0      # montée max ≈ 3 °C/min
_ramp_down_K_per_min = 30.0   # descente max ≈ 30 °C/min
_ramp_flow_kgps_per_min = 6.0 # variation débit ≈ 6 kg/s par minute

CONTROL_LIMITS = dict(
    temp_min=50.0,
    temp_max=110.0,
    flow_min=3.0,
    flow_max=30.0,
    max_temp_rise_per_dt=_ramp_up_K_per_min * _dt / 60.0,
    max_temp_drop_per_dt=_ramp_down_K_per_min * _dt / 60.0,
    max_flow_delta_per_dt=_ramp_flow_kgps_per_min * _dt / 60.0,
)

_episode_length_steps = int(SIMULATION_PARAMS["t_max_day"] / SIMULATION_PARAMS["dt"])
_total_episodes = 10
_total_timesteps = _episode_length_steps * _total_episodes
# --- Paramètres d'entraînement RL ---
RL_TRAINING = dict(
    total_timesteps=_total_timesteps,
    episode_length_steps=_episode_length_steps,
    n_steps_update=200,
    learning_rate=3e-4,
    save_freq_episodes=1,
    use_s3_checkpoints=False,  # False = uniquement local, True = local + S3
)

# Ancien réseau (env 5MW)
# EDGES = [
#     (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
#     (8, 20),
#     (7, 18), (18, 19),
#     (6, 15), (15, 16),
#     (5, 13), (13, 14),
#     (4, 23), (23, 24),
#     (3, 10), (10, 11),
# ]
