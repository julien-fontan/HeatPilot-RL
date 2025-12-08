import numpy as np
from scipy.interpolate import interp1d
from dataclasses import dataclass

def generate_step_function(t_end, step_time, min_val, max_val, seed):
    """
    Génère une fonction temporelle en ESCALIER (ancienne version).
    - step_time : durée entre deux changements de valeur (s).
    """
    rng = np.random.default_rng(seed)
    time_knots = np.arange(0.0, t_end + step_time, step_time)
    values = rng.uniform(min_val, max_val, size=len(time_knots))

    interpolator = interp1d(
        time_knots,
        values,
        kind="previous",
        bounds_error=False,
        fill_value=(values[0], values[-1]),
    )
    return lambda t: float(interpolator(t))

def generate_smooth_profile(t_end, step_time, min_val, max_val, seed, smooth_factor: float = 1.0):
    """
    Génère une fonction temporelle aléatoire LISSEE (plus ou moins "fluide").

    Esprit:
      - on génère un bruit blanc aléatoire discret sur [0, t_end],
      - on le lisse par convolution sur une fenêtre temporelle,
      - on le renormalise entre [min_val, max_val],
      - on interpole ensuite linéairement pour obtenir une fonction continue.

    Paramètres:
      - t_end       : durée totale (s),
      - step_time   : échelle de temps "macro" (s) utilisée comme base pour l'échantillonnage,
      - smooth_factor :
            = 1.0  : lissage modéré,
          > 1.0    : profils plus lisses (fenêtre de lissage plus large).
    """
    rng = np.random.default_rng(seed)

    # Pas de base pour l'échantillonnage du bruit brut
    dt_base = step_time
    t_samples = np.arange(0.0, t_end + dt_base, dt_base)

    # Bruit blanc uniforme brut
    raw = rng.uniform(low=-1.0, high=1.0, size=len(t_samples))

    # Fenêtre de lissage (convolution) -> plus smooth_factor est grand, plus c'est lisse
    # largeur_effective ≈ smooth_factor * step_time
    window_duration = smooth_factor * step_time
    n_win = max(3, int(window_duration / dt_base))  # au moins 3 points
    if n_win % 2 == 0:
        n_win += 1  # fenêtre impaire

    window = np.ones(n_win, dtype=float)
    window /= window.sum()

    # Convolution 'same' pour lisser le signal
    smoothed = np.convolve(raw, window, mode="same")

    # Normalisation dans [0, 1]
    sm_min, sm_max = smoothed.min(), smoothed.max()
    if sm_max > sm_min:
        smoothed_norm = (smoothed - sm_min) / (sm_max - sm_min)
    else:
        smoothed_norm = np.zeros_like(smoothed)

    # Remise à l'échelle [min_val, max_val]
    values = min_val + smoothed_norm * (max_val - min_val)

    # Interpolation linéaire continue
    interpolator = interp1d(
        t_samples,
        values,
        kind="linear",
        bounds_error=False,
        fill_value=(values[0], values[-1]),
    )
    return lambda t: float(interpolator(t))