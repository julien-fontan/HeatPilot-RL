"""
Configuration centralisée pour le réseau de chaleur et l'entraînement RL.
Contient uniquement des paramètres considérés "fixes" pour une campagne donnée.
"""

# --- Seed globale pour génération de paramètres aléatoires ---
GLOBAL_SEED = 42

# --- Topologie globale ---
EDGES = [
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
    (3,7), (7,8),
    (5,9), (9,10),(10,11),
]

# Fractionnements par défaut pour les noeuds de branchement
DEFAULT_NODE_SPLITS = {3: {7: 0.3, 4: 0.7}, 5: {6: 0.3, 9: 0.7}}

# --- Paramètres physiques ---
PHYSICAL_PROPS = dict(
    rho=1000.0,
    cp=4182.0,
    thermal_conductivity=0.0,
    external_temp=10.0,
    # Paramètres de génération des conduites
    length_min=50.0,                    # m
    length_max=300.0,
    diameter_min=0.15,                  # m
    diameter_max=0.35,
    heat_loss_coeff_min=0.90,           # W/m²/K
    heat_loss_coeff_max=1.75,
)

# --- Paramètres de discrétisation, conditions initiales de la simulation ---
# température minimale de retour (utilisée pour le soutirage de puissance et l'état initial)
_min_return_temp = 40.0                 # °C

SIMULATION_PARAMS = dict(
    dx=10,                              # m, influence énormément la durée de la simulation
    t_max_day=24 * 3600.0,              # s (1 épisode simulé = 1 journée)
    rtol=1e-4,
    atol=1e-4,
    warmup_duration=3600*3,             # Durée de pré-chauffe (warmup) avant la simulation principale (s)
    min_return_temp=_min_return_temp,
    initial_temp=75,                    # °C
    initial_flow=15.0,                  # kg/s
    randomize_geometry=False            # Si True, régénère les longueurs/diamètres à chaque reset
)

# --- Pas de contrôle RL ---
dt_rl = 60  # s

# --- Paramètres des profils de puissance demandée par les consommateurs ---
POWER_PROFILE_CONFIG = dict(
    p_min=100_000.0,                    # W
    p_max=300_000.0,                    # W
    step_time=3600.0,                   # s entre deux changements de demande
    smooth_factor=1.0,                  # >1 => profils plus lisses
)

# nombre de pas RL entre deux changements de demande, pour information
POWER_PROFILE_CONFIG["step_time_steps"] = int(POWER_PROFILE_CONFIG["step_time"] / dt_rl)

# --- Contraintes physiques / actionneurs (utilisées dans l'env Gym) ---
# rampes exprimées en K/min et kg/s/min puis converties en "par pas" via dt_rl
_ramp_up_K_per_min = 3.0                # montée max : 3 °C/min
_ramp_down_K_per_min = 3.0              # en pratique, en mélangeant le fluide avec la sortie : 20 °C/min
_ramp_flow_kgps_per_min = 3.0           # variation débit : 3 kg/s par minute
_ramp_split_per_min = 0.1               # variation split : 10% par minute

CONTROL_PARAMS = dict(
    temp_min=50.0,
    temp_max=120.0,
    flow_min=3.0,
    flow_max=30.0,
    max_temp_rise_per_dt=_ramp_up_K_per_min * dt_rl / 60.0,
    max_temp_drop_per_dt=_ramp_down_K_per_min * dt_rl / 60.0,
    max_flow_delta_per_dt=_ramp_flow_kgps_per_min * dt_rl / 60.0,
    max_split_change_per_dt=_ramp_split_per_min * dt_rl / 60.0,
    enable_ramps=True,                  # True = applique les rampes physiques, False = changement instantané
)

# --- Paramètres d'entraînement RL ---
_episode_length_steps = int(SIMULATION_PARAMS["t_max_day"] / dt_rl)
_total_episodes = 3000                  # nombre total d'épisodes d'entraînement
_total_timesteps = _episode_length_steps * _total_episodes
_n_steps_update = int(24*3600 / dt_rl)  # = nb_heures de simulation avant update

TRAINING_PARAMS = dict(
    dt=dt_rl,
    total_timesteps=_total_timesteps,
    episode_length_steps=_episode_length_steps,
    n_steps_update=_n_steps_update,
    batch_size=int(_n_steps_update/4),
    learning_rate=5e-5,
    ent_coef=5e-4,
    save_freq_episodes=100,
    gamma=0.999,
    warmup_enabled=True,                # si True, utilise une phase de warmup avant chaque épisode
)

# --- Poids de la fonction de récompense ---
REWARD_PARAMS = dict(
    weights=dict(
        comfort=4,
        waste=2,
        pump=5,
    ),
    params=dict(
        p_ref=2000.0,                   # Puissance de référence (kW)
        p_pump_nominal=15.0,            # Puissance nominale de la pompe (kW)
        p_pump_sigma=5.0,               # Largeur du bonus pompe (kg/s) : plus petit = plus strict autout de p_pump_nominal

        # Confort : bonus quand la puissance fournie est proche de la demande
        supply_bonus_amp=2.0,           # Amplitude du bonus confort
        sigma_supply=100.0,             # Largeur du bonus confort (kW)

        # Sobriété : bonus quand les pertes sont faibles
        waste_bonus_amp=1.0,
        sigma_waste=100.0,

        # Bonus combo : gros bonus si confort OK ET pertes faibles
        combo_bonus=10.0,               # Bonus
        combo_comfort_deficit_max_kw=15.0, # Confort atteint si (dem_kw - sup_kw) <= ce seuil
        combo_wasted_max_kw=25.0,       # Pertes faibles si wasted_kw <= ce seuil
    )
)