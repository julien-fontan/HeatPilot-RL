import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from district_heating_gym_env import HeatNetworkEnv
from graph_visualization import DistrictHeatingVisualizer
import config

# --- CONFIGURATION PAR DÉFAUT ---
DEFAULT_MODEL_SUBDIR = "PPO_12"  # Nom du dossier par défaut
DEFAULT_ITER = 1900800                   # None = prend le dernier checkpoint
SMOOTHING_WINDOW_MIN = 3.0           # Fenêtre de lissage (minutes) pour les graphiques
# --------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def find_model_path(subdir, iteration=None):
    """
    Localise intelligemment le fichier modèle et sa normalisation.
    """
    run_dir = os.path.join(MODELS_ROOT_DIR, subdir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Dossier du run introuvable : {run_dir}")

    # Si une itération spécifique est demandée
    if iteration is not None:
        model_name = f"{subdir}_{iteration}"
        model_path = os.path.join(run_dir, model_name)
        if not os.path.exists(model_path + ".zip"):
             raise FileNotFoundError(f"Checkpoint introuvable : {model_name}.zip")
        return model_path, iteration

    # Sinon, on cherche le dernier
    candidates = []
    prefix = f"{subdir}_"
    for fname in os.listdir(run_dir):
        if fname.startswith(prefix) and fname.endswith(".zip"):
            try:
                # Extraction du numéro après le dernier "_" et avant ".zip"
                suffix = fname[len(prefix):-4]
                candidates.append(int(suffix))
            except ValueError:
                continue
    
    if not candidates:
        raise FileNotFoundError(f"Aucun checkpoint valide trouvé dans {run_dir}")
    
    best_iter = max(candidates)
    print(f"-> Dernier checkpoint trouvé : itération {best_iter}")
    return os.path.join(run_dir, f"{subdir}_{best_iter}"), best_iter

def smooth_signal(data, dt, window_min):
    """Applique une moyenne mobile pour lisser les courbes."""
    if window_min <= 0: return data
    window_sec = window_min * 60.0
    window_size = int(window_sec / dt)
    if window_size < 2: return data
    
    kernel = np.ones(window_size) / window_size
    # mode='same' garde la même taille de vecteur
    return np.convolve(data, kernel, mode='same')

def run_evaluation(subdir, iteration_req=None, force_rerun=True):
    # 1. Localisation des fichiers
    try:
        model_path, iteration = find_model_path(subdir, iteration_req)
    except FileNotFoundError as e:
        print(f"[ERREUR] {e}")
        return

    run_dir = os.path.dirname(model_path)
    run_id = f"{subdir}_{iteration}"
    
    # 2. Chargement Environnement & Normalisation
    print(f"Chargement de l'environnement pour {run_id}...")
    env = HeatNetworkEnv()
    
    # On force la géométrie fixe pour l'évaluation (évite la recréation inutile)
    env.randomize_geometry = False 
    
    env = DummyVecEnv([lambda: env])
    
    # Chargement des stats de normalisation (critique pour PPO)
    norm_path = os.path.join(run_dir, f"vec_normalize_{subdir}_{iteration}.pkl")
    if os.path.exists(norm_path):
        print(f"Chargement VecNormalize : {os.path.basename(norm_path)}")
        env = VecNormalize.load(norm_path, env)
        env.training = False     # Ne pas mettre à jour les stats
        env.norm_reward = False  # On veut voir les vraies rewards
    else:
        print("ATTENTION : Pas de fichier VecNormalize trouvé. Les résultats seront probablement incorrects.")

    # 3. Chargement Agent
    print(f"Chargement du modèle PPO...")
    model = PPO.load(model_path)

    # 4. Simulation
    real_env = env.envs[0].unwrapped
    obs = env.reset()
    
    # Préparation des structures de données pour l'enregistrement
    consumer_nodes = real_env.consumer_nodes
    topo_parents = real_env.graph.get_parent_nodes()
    
    # Mapping optimisé : Pour chaque consommateur, quels sont les indices des tuyaux entrants ?
    incoming_pipes_indices = {}
    for nid in consumer_nodes:
        incoming_pipes_indices[nid] = []
        for p_node in topo_parents.get(nid, []):
            edge = real_env.graph.edges[(p_node, nid)]
            incoming_pipes_indices[nid].append(edge["pipe_index"])

    # Historique
    hist = {
        "time": [], 
        "T_source": [], "m_source": [],
        "demand_total": [], "supplied_total": [], "wasted_total": [],
        "nodes": {nid: {"T_in": [], "m_in": [], "p_dem": [], "p_sup": []} for nid in consumer_nodes}
    }
    
    # Constantes physiques locales pour vérification
    cp = real_env.props["cp"]
    min_ret_temp = config.SIMULATION_PARAMS["min_return_temp"]

    print("Exécution de la simulation...")
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        if dones[0]:
            break
        
        # Extraction des données brutes de physique
        info = infos[0]
        t = real_env.current_t
        pipe_flows = info["pipe_mass_flows"]
        node_temps = info["node_temperatures"]

        # Enregistrement Global
        hist["time"].append(t)
        hist["T_source"].append(real_env.actual_inlet_temp)
        hist["m_source"].append(real_env.actual_mass_flow)
        
        step_demand = 0.0
        step_supplied = 0.0
        step_wasted = 0.0
        
        # Enregistrement par Nœud (Recalcul pour validation)
        for nid in consumer_nodes:
            # 1. Débit entrant total au noeud
            m_in = sum(pipe_flows[pidx] for pidx in incoming_pipes_indices[nid])
            
            # 2. Température au noeud
            node_idx = real_env.graph.get_id_from_node(nid)
            T_in = node_temps[node_idx]
            
            # 3. Demande cible
            p_dem = real_env.demand_funcs[nid](t)
            
            # 4. Calcul de ce qui est physiquement possible de fournir
            delta_T_avail = max(T_in - min_ret_temp, 0.0)
            p_phys_max = m_in * cp * delta_T_avail
            
            p_sup = min(p_dem, p_phys_max)
            p_wasted = max(0.0, p_phys_max - p_sup) # Énergie disponible mais non consommée (car T_in trop haute pour rien)

            hist["nodes"][nid]["T_in"].append(T_in)
            hist["nodes"][nid]["m_in"].append(m_in)
            hist["nodes"][nid]["p_dem"].append(p_dem)
            hist["nodes"][nid]["p_sup"].append(p_sup)
            
            step_demand += p_dem
            step_supplied += p_sup
            step_wasted += p_wasted
            
        hist["demand_total"].append(step_demand)
        hist["supplied_total"].append(step_supplied)
        hist["wasted_total"].append(step_wasted)

    print("Simulation terminée. Génération des graphiques...")
    
    # 5. Post-traitement et Lissage
    sim_dt = real_env.dt
    t_arr = np.array(hist["time"])
    
    def proc(data_list):
        arr = np.array(data_list)
        return smooth_signal(arr, sim_dt, SMOOTHING_WINDOW_MIN)

    # Conversion au format attendu par le Visualizer
    final_data = {
        "time": t_arr,
        "T_source": proc(hist["T_source"]),
        "m_source": proc(hist["m_source"]),
        "demand_total": proc(hist["demand_total"]),
        "supplied_total": proc(hist["supplied_total"]),
        "wasted_total": proc(hist["wasted_total"]),
        "node_ids": np.array(consumer_nodes),
    }
    for nid in consumer_nodes:
        for k in ["T_in", "m_in", "p_dem", "p_sup"]:
            final_data[f"node_{nid}_{k}"] = proc(hist["nodes"][nid][k])

    # 6. Visualisation
    visualizer = DistrictHeatingVisualizer(PLOTS_DIR)
    visualizer.plot_dashboard_general(final_data, title_suffix=run_id, )
    visualizer.plot_dashboard_nodes_2cols(final_data, title_suffix=run_id)
    
    print("Terminé.")

if __name__ == "__main__":
    # Petit parser CLI pour faciliter l'usage
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=DEFAULT_MODEL_SUBDIR, help="Nom du dossier dans models/")
    parser.add_argument("--iter", type=int, default=DEFAULT_ITER, help="Numéro de l'itération (optionnel)")
    args = parser.parse_args()

    run_evaluation(args.run, args.iter)