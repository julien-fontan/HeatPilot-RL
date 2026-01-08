import numpy as np
from scipy.interpolate import interp1d

def generate_step_function(t_end, step_time, min_val, max_val, seed):
    rng = np.random.default_rng(seed)
    time_knots = np.arange(0.0, t_end + step_time, step_time)
    values = rng.uniform(min_val, max_val, size=len(time_knots))
    interpolator = interp1d(time_knots, values, kind="previous", bounds_error=False, fill_value=(values[0], values[-1]))
    return lambda t: float(interpolator(t))

def generate_smooth_profile(t_end, step_time, min_val, max_val, seed, smooth_factor: float = 1.0):
    rng = np.random.default_rng(seed)
    dt_base = step_time
    t_samples = np.arange(0.0, t_end + dt_base, dt_base)
    raw = rng.uniform(low=-1.0, high=1.0, size=len(t_samples))

    window_duration = smooth_factor * step_time
    n_win = max(3, int(window_duration / dt_base))
    if n_win % 2 == 0: n_win += 1
    
    window = np.ones(n_win, dtype=float) / float(n_win)
    smoothed = np.convolve(raw, window, mode="same")

    sm_min, sm_max = smoothed.min(), smoothed.max()
    if sm_max > sm_min: smoothed_norm = (smoothed - sm_min) / (sm_max - sm_min)
    else: smoothed_norm = np.zeros_like(smoothed)

    values = min_val + smoothed_norm * (max_val - min_val)
    interpolator = interp1d(t_samples, values, kind="linear", bounds_error=False, fill_value=(values[0], values[-1]))
    return lambda t: float(interpolator(t))