import numpy as np
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from district_heating_gym_env import HeatNetworkEnv
from graph_visualization import DistrictHeatingVisualizer
import config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

window_min = 8.0  # minutes pour le lissage (moyenne mobile) des signaux lors de l'évaluation

def find_model_path(subdir, iteration=None):
    run_dir = os.path.join(MODELS_ROOT_DIR, subdir)
    if not os.path.isdir(run_dir): raise FileNotFoundError(f"Introuvable: {run_dir}")

    if iteration is not None:
        path = os.path.join(run_dir, f"{subdir}_{iteration}")
        if not os.path.exists(path + ".zip"): raise FileNotFoundError(f"Checkpoint {iteration} introuvable")
        return path, iteration

    candidates = []
    for fname in os.listdir(run_dir):
        if fname.startswith(f"{subdir}_") and fname.endswith(".zip"):
            try: candidates.append(int(fname[len(subdir)+1:-4]))
            except: continue
    if not candidates: raise FileNotFoundError("Aucun checkpoint trouvé")
    best = max(candidates)
    return os.path.join(run_dir, f"{subdir}_{best}"), best

def smooth_signal(data, dt, window_min):
    if window_min <= 0: return np.array(data)
    window = int(window_min * 60.0 / dt)
    if window < 2: return np.array(data)
    kernel = np.ones(window) / window
    padded = np.pad(data, (window//2, window-1-window//2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")

def run_evaluation(subdir, iteration_req=None, use_heuristic_splits=False):
    model_path, iteration = find_model_path(subdir, iteration_req)
    run_id = f"{subdir}_{iteration}"
    suffix = "_heuristic" if use_heuristic_splits else "_agent"
    output_path = os.path.join(os.path.dirname(model_path), f"eval_data_{run_id}{suffix}.npz")

    if os.path.exists(output_path):
        print(f"Chargement données existantes: {output_path}")
        loaded = np.load(output_path, allow_pickle=True)
        final_data = {key: loaded[key] for key in loaded.files}
    else:
        print("Lancement simulation...")
        env = HeatNetworkEnv(fixed_seed=config.GLOBAL_SEED)
        env.randomize_geometry = False
        env = DummyVecEnv([lambda: env])
        
        real_env = env.envs[0].unwrapped
        if use_heuristic_splits:
            print("[MODE] HEURISTIQUE (Vannes auto)")
            real_env.set_agent_split_control(False)
        else:
            print("[MODE] AGENT (Vannes IA)")
            real_env.set_agent_split_control(True)

        norm_path = os.path.join(os.path.dirname(model_path), f"vec_normalize_{subdir}_{iteration}.pkl")
        if os.path.exists(norm_path):
            env = VecNormalize.load(norm_path, env)
            env.training = False
            env.norm_reward = False

        model = PPO.load(model_path)
        obs = env.reset()
        
        hist = {"time": [], "T_src": [], "m_src": [], "splits": [], 
                "dem": [], "sup": [], "wast": [],
                "nodes": {n: {"T": [], "m": [], "dem": [], "sup": []} for n in real_env.graph.consumer_nodes}}

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action)
            info = infos[0]
            if dones[0]: break
            
            t = real_env.current_t
            hist["time"].append(t)
            hist["T_src"].append(real_env.actual_inlet_temp)
            hist["m_src"].append(real_env.actual_mass_flow)
            hist["splits"].append(real_env.current_split_ratios.copy())
            
            metrics = real_env.network.get_instant_metrics(t, real_env.state)
            hist["dem"].append(metrics["demand"])
            hist["sup"].append(metrics["supplied"])
            hist["wast"].append(metrics["wasted"])

            for nid in real_env.graph.consumer_nodes:
                idx = real_env.graph.get_id(nid)
                hist["nodes"][nid]["T"].append(metrics["node_temperatures"][idx])
                hist["nodes"][nid]["dem"].append(real_env.demand_funcs[nid](t))
                
                p_dem = real_env.demand_funcs[nid](t)
                m_in = 0.0
                for _, pidx in real_env.graph.int_parent_adjacency[idx]:
                    m_in += metrics["mass_flows"][pidx]
                
                avail = m_in * real_env.props["cp"] * max(metrics["node_temperatures"][idx] - config.SIMULATION_PARAMS["min_return_temp"], 0)
                p_sup = min(p_dem, avail)
                
                hist["nodes"][nid]["m"].append(m_in)
                hist["nodes"][nid]["sup"].append(p_sup)

        final_data = {
            "time": np.array(hist["time"]),
            "T_source": smooth_signal(hist["T_src"], real_env.dt, window_min),
            "m_source": smooth_signal(hist["m_src"], real_env.dt, window_min),
            "demand_total": smooth_signal(hist["dem"], real_env.dt, window_min),
            "supplied_total": smooth_signal(hist["sup"], real_env.dt, window_min),
            "wasted_total": smooth_signal(hist["wast"], real_env.dt, window_min),
            "node_ids": np.array(real_env.graph.consumer_nodes)
        }
        
        splits_arr = np.array(hist["splits"])
        for i, nid in enumerate(real_env.branching_nodes):
            final_data[f"split_node_{nid}"] = smooth_signal(splits_arr[:, i], real_env.dt, window_min)
            
        for nid in real_env.graph.consumer_nodes:
            final_data[f"node_{nid}_T_in"] = smooth_signal(hist["nodes"][nid]["T"], real_env.dt, window_min)
            final_data[f"node_{nid}_m_in"] = smooth_signal(hist["nodes"][nid]["m"], real_env.dt, window_min)
            final_data[f"node_{nid}_p_dem"] = smooth_signal(hist["nodes"][nid]["dem"], real_env.dt, window_min)
            final_data[f"node_{nid}_p_sup"] = smooth_signal(hist["nodes"][nid]["sup"], real_env.dt, window_min)

        np.savez_compressed(output_path, **final_data)

    print("Génération graphiques...")
    viz = DistrictHeatingVisualizer(PLOTS_DIR)
    viz.plot_dashboard_general(final_data, title_suffix=f"{run_id}{suffix}")
    viz.plot_dashboard_nodes_2cols(final_data, title_suffix=f"{run_id}{suffix}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="PPO_26")
    parser.add_argument("--iter", type=int, default=None)
    parser.add_argument("--heuristic-splits", action="store_true")
    args = parser.parse_args()

    run_evaluation(args.run, args.iter, args.heuristic_splits)