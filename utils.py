import numpy as np
from scipy.interpolate import interp1d
from dataclasses import dataclass

@dataclass
class PhysicalProperties:
    rho: float = 1000.0
    cp: float = 4182.0
    thermal_conductivity: float = 0.0
    external_temp: float = 10.0
    heat_loss_coeff: float = 0.0

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