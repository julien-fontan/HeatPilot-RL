import numpy as np
from scipy.interpolate import interp1d

def generate_step_function(t_end, step_time, min_val, max_val, seed):
    """
    Génère une fonction temporelle en escalier (valeurs aléatoires).
    """
    rng = np.random.default_rng(seed)
    time_knots = np.arange(0.0, t_end + step_time, step_time)
    
    values = rng.uniform(min_val, max_val, size=len(time_knots))
    
    # Interpolation "previous" : maintient la valeur précédente jusqu'au prochain changement
    interpolator = interp1d(
        time_knots, values, 
        kind="previous", 
        bounds_error=False, 
        fill_value=(values[0], values[-1])
    )
    
    # On retourne une lambda pour avoir une interface simple f(t) -> float
    return lambda t: float(interpolator(t))