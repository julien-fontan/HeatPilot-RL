import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from district_heating_gym_env import HeatNetworkEnv
from graph_visualization import DistrictHeatingVisualizer
import config

# --- CONFIGURATION PAR DÉFAUT ---actuellement
DEFAULT_MODEL_SUBDIR = "PPO_26"  # Nom du dossier par défaut
DEFAULT_ITER = None              # None = prend le dernier checkpoint
SMOOTHING_WINDOW_MIN = 10.0       # Fenêtre de lissage (minutes) pour les graphiques
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
    arr = np.asarray(data)
    if window_min <= 0:
        return arr

    if dt <= 0:
        raise ValueError(f"dt doit être > 0 (reçu: {dt})")

    window_sec = window_min * 60.0
    window_size = int(window_sec / dt)
    if window_size < 2 or arr.size == 0:
        return arr

    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)

    # IMPORTANT:
    # np.convolve(..., mode='same') implique un padding par zéros aux bords,
    # ce qui tire artificiellement les premières/dernières valeurs vers 0.
    # On pad explicitement avec les valeurs de bord pour éviter cet artefact.
    pad_left = window_size // 2
    pad_right = window_size - 1 - pad_left
    padded = np.pad(arr.astype(np.float64, copy=False), (pad_left, pad_right), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed.astype(arr.dtype, copy=False)

def run_evaluation(subdir, iteration_req=None, force_rerun=False, use_heuristic_splits=False):
    # 1. Localisation des fichiers (Modèle & Data)
    try:
        model_path, iteration = find_model_path(subdir, iteration_req)
    except FileNotFoundError as e:
        print(f"[ERREUR] {e}")
        return

    run_dir = os.path.dirname(model_path)
    run_id = f"{subdir}_{iteration}"
    
    # On ajoute un suffixe au fichier cache si on utilise l'heuristique
    # pour ne pas écraser les résultats de l'agent complet
    suffix_mode = "_heuristic" if use_heuristic_splits else "_agent"
    output_filename = f"eval_data_{run_id}{suffix_mode}.npz"
    output_path = os.path.join(run_dir, output_filename)
    
    final_data = None

    # ---------------------------------------------------------
    # 2. VÉRIFICATION DU CACHE (FICHIER .NPZ)
    # ---------------------------------------------------------
    if os.path.exists(output_path) and not force_rerun:
        print(f"\n[INFO] Fichier de données existant trouvé : {output_filename}")
        print("[INFO] Chargement direct sans relancer la simulation...")
        
        # Chargement du fichier compressé
        loaded = np.load(output_path, allow_pickle=True)
        # Conversion en dictionnaire standard pour compatibilité
        final_data = {key: loaded[key] for key in loaded.files}
        
    else:
        if force_rerun:
            print(f"\n[INFO] Force rerun activé. Nouvelle simulation...")
        else:
            print(f"\n[INFO] Pas de fichier de données trouvé. Lancement de la simulation...")

        # ---------------------------------------------------------
        # 3. SIMULATION (SI NÉCESSAIRE)
        # ---------------------------------------------------------
        
        # a. Chargement Environnement & Normalisation
        env = HeatNetworkEnv()
        env.randomize_geometry = False # Fixe la géométrie
        env = DummyVecEnv([lambda: env])
        
        # --- CONFIGURATION DU MODE DE CONTRÔLE DES VANNES ---
        # On accède à l'environnement interne (HeatNetworkEnv)
        real_env_unwrapped = env.envs[0].unwrapped
        if hasattr(real_env_unwrapped, "set_agent_split_control"):
            # Si use_heuristic_splits est True, on DÉSACTIVE le contrôle agent (False)
            agent_control = not use_heuristic_splits
            real_env_unwrapped.set_agent_split_control(agent_control)
            mode_str = "HEURISTIQUE (Automatique)" if use_heuristic_splits else "AGENT (Appris)"
            print(f"\n[EVAL CONFIG] Mode de gestion des vannes : {mode_str}")
        else:
            print("\n[WARN] Impossible de configurer le contrôle des vannes (méthode manquante dans l'env).")

        # b. Chargement Normalisation
        norm_path = os.path.join(run_dir, f"vec_normalize_{subdir}_{iteration}.pkl")
        if os.path.exists(norm_path):
            env = VecNormalize.load(norm_path, env)
            env.training = False     
            env.norm_reward = False  
        else:
            print("ATTENTION : Pas de fichier VecNormalize trouvé.")

        # c. Chargement Agent
        print(f"Chargement du modèle PPO...")
        model = PPO.load(model_path)

        # d. Boucle de Simulation
        real_env = env.envs[0].unwrapped # Récupération après chargement éventuel
        obs = env.reset()
        
        consumer_nodes = real_env.consumer_nodes
        topo_parents = real_env.graph.get_parent_nodes()
        
        incoming_pipes_indices = {}
        for nid in consumer_nodes:
            incoming_pipes_indices[nid] = []
            for p_node in topo_parents.get(nid, []):
                edge = real_env.graph.edges[(p_node, nid)]
                incoming_pipes_indices[nid].append(edge["pipe_index"])

        hist = {
            "time": [], 
            "T_source": [], "m_source": [],
            "demand_total": [], "supplied_total": [], "wasted_total": [],
            "nodes": {nid: {"T_in": [], "m_in": [], "p_dem": [], "p_sup": []} for nid in consumer_nodes},
            "splits":[]
        }
        
        cp = real_env.props["cp"]
        min_ret_temp = config.SIMULATION_PARAMS["min_return_temp"]
        terminal_nodes = real_env.graph.get_terminal_nodes()

        print("Exécution de la simulation en cours...")
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            
            if dones[0]:
                break
            
            info = infos[0]
            t = real_env.current_t
            pipe_flows = info["pipe_mass_flows"]
            node_temps = info["node_temperatures"]
            # Températures après soutirage aux sous-stations (utile pour calculer le "wasted" terminal)
            node_temps_after = real_env.network._apply_node_power_consumption(t, node_temps, pipe_flows)

            hist["time"].append(t)
            hist["T_source"].append(real_env.actual_inlet_temp)
            hist["m_source"].append(real_env.actual_mass_flow)
            hist["splits"].append(real_env.current_split_ratios.copy())
            
            step_demand = 0.0
            step_supplied = 0.0
            step_wasted = 0.0
            
            for nid in consumer_nodes:
                m_in = sum(pipe_flows[pidx] for pidx in incoming_pipes_indices[nid])
                node_idx = real_env.graph.get_id_from_node(nid)
                T_in = node_temps[node_idx]
                p_dem = real_env.demand_funcs[nid](t)
                
                delta_T_avail = max(T_in - min_ret_temp, 0.0)
                p_phys_max = m_in * cp * delta_T_avail
                
                p_sup = min(p_dem, p_phys_max)

                hist["nodes"][nid]["T_in"].append(T_in)
                hist["nodes"][nid]["m_in"].append(m_in)
                hist["nodes"][nid]["p_dem"].append(p_dem)
                hist["nodes"][nid]["p_sup"].append(p_sup)
                
                step_demand += p_dem
                step_supplied += p_sup

            # "Wasted" cohérent avec l'env: pertes thermiques inutiles aux noeuds terminaux
            # (si T_out après consommation est au-dessus de la température mini de retour)
            for nid in terminal_nodes:
                node_idx = real_env.graph.get_id_from_node(nid)
                T_out = node_temps_after[node_idx]

                # Débit entrant au terminal (somme des débits des conduites entrantes)
                m_in = 0.0
                for p_node in topo_parents.get(nid, []):
                    edge = real_env.graph.edges[(p_node, nid)]
                    m_in += pipe_flows[edge["pipe_index"]]

                if m_in > 1e-9 and T_out > min_ret_temp:
                    step_wasted += m_in * cp * (T_out - min_ret_temp)
                
            hist["demand_total"].append(step_demand)
            hist["supplied_total"].append(step_supplied)
            hist["wasted_total"].append(step_wasted)

        print("Simulation terminée. Traitement des données...")
        
        # e. Post-traitement et Lissage
        sim_dt = real_env.dt
        t_arr = np.array(hist["time"])
        
        def proc(data_list):
            arr = np.array(data_list)
            return smooth_signal(arr, sim_dt, SMOOTHING_WINDOW_MIN)

        final_data = {
            "time": t_arr,
            "T_source": proc(hist["T_source"]),
            "m_source": proc(hist["m_source"]),
            "demand_total": proc(hist["demand_total"]),
            "supplied_total": proc(hist["supplied_total"]),
            "wasted_total": proc(hist["wasted_total"]),
            "node_ids": np.array(consumer_nodes),
        }

        splits_arr = np.array(hist["splits"])
        for i, node_id in enumerate(real_env.branching_nodes):
                key_name = f"split_node_{node_id}"
                final_data[key_name] = proc(splits_arr[:, i])

        for nid in consumer_nodes:
            for k in ["T_in", "m_in", "p_dem", "p_sup"]:
                final_data[f"node_{nid}_{k}"] = proc(hist["nodes"][nid][k])

        # f. Sauvegarde
        print(f"Sauvegarde des données dans : {output_path}")
        np.savez_compressed(output_path, **final_data)

    # ---------------------------------------------------------
    # 4. VISUALISATION
    # ---------------------------------------------------------
    print("Génération des graphiques...")
    
    # On ajoute le mode au titre du graphique pour s'y retrouver
    mode_suffix = "_HEURISTIC" if use_heuristic_splits else "_AGENT"
    title_suffix = f"{run_id}{mode_suffix}"
    
    visualizer = DistrictHeatingVisualizer(PLOTS_DIR)
    visualizer.plot_dashboard_general(final_data, title_suffix=title_suffix)
    visualizer.plot_dashboard_nodes_2cols(final_data, title_suffix=title_suffix)
    
    print("Terminé.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=DEFAULT_MODEL_SUBDIR, help="Nom du dossier dans models/")
    parser.add_argument("--iter", type=int, default=DEFAULT_ITER, help="Numéro de l'itération (optionnel)")
    parser.add_argument("--force", action="store_true", help="Forcer la ré-exécution de la simulation même si le fichier .npz existe")
    
    # NOUVEAU FLAG
    parser.add_argument("--heuristic-splits", action="store_true", help="Désactive l'agent pour les splits et utilise l'heuristique idéale.")
    
    args = parser.parse_args()

    run_evaluation(
        args.run, 
        args.iter, 
        force_rerun=args.force, 
        use_heuristic_splits=args.heuristic_splits
    )