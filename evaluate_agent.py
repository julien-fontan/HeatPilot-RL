import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from district_heating_gym_env import HeatNetworkEnv
import config

# --- CONFIGURATION UTILISATEUR ---
MODEL_SUBDIR = "PPO_test7_rampes"  # Nom du dossier dans 'models'
MODEL_ITER = None                  # None = prend le dernier checkpoint, ou un entier (ex: 3456000)
# ---------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_ROOT_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def find_model_path(subdir, iteration=None):
    run_dir = os.path.join(MODELS_ROOT_DIR, subdir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Dossier introuvable : {run_dir}")

    if iteration is not None:
        model_name = f"{subdir}_{iteration}"
        model_path = os.path.join(run_dir, model_name)
        if not os.path.exists(model_path + ".zip"):
             raise FileNotFoundError(f"Modèle introuvable : {model_path}.zip")
        return model_path, iteration

    candidates = []
    prefix = f"{subdir}_"
    for fname in os.listdir(run_dir):
        if fname.startswith(prefix) and fname.endswith(".zip"):
            try:
                suffix = fname[len(prefix):-4]
                candidates.append(int(suffix))
            except ValueError: continue
    
    if not candidates:
        raise FileNotFoundError(f"Aucun modèle dans {run_dir}")
    
    best_iter = max(candidates)
    return os.path.join(run_dir, f"{subdir}_{best_iter}"), best_iter

def main():
    # 1. Chargement du modèle
    if not MODEL_SUBDIR:
        print("Erreur: MODEL_SUBDIR requis.")
        return

    try:
        model_path, iteration = find_model_path(MODEL_SUBDIR, MODEL_ITER)
        print(f" Chargement : {model_path}.zip")
    except FileNotFoundError as e:
        print(e)
        return

    run_dir = os.path.dirname(model_path)

    # 2. Préparation de l'environnement
    env = HeatNetworkEnv()
    env = DummyVecEnv([lambda: env]) # Encapsulation standard SB3

    # Chargement des statistiques de normalisation (CRUCIAL)
    norm_path = os.path.join(run_dir, f"vec_normalize_{MODEL_SUBDIR}_{iteration}.pkl")
    if os.path.exists(norm_path):
        print(f" Chargement VecNormalize : {os.path.basename(norm_path)}")
        env = VecNormalize.load(norm_path, env)
        env.training = False     # Mode test : on ne met plus à jour les moyennes
        env.norm_reward = False  # On veut voir les rewards réels
    else:
        print(" ATTENTION : Pas de fichier VecNormalize trouvé. Les résultats risquent d'être incohérents.")

    model = PPO.load(model_path)
    
    # On récupère l'environnement physique pour accéder à la topologie statique
    real_env = env.envs[0].unwrapped
    
    # 3. Structures de données pour l'historique
    history = {
        "time": [], "T_inlet": [], "m_inlet": [], 
        "demand_total": [], "supplied_total": []
    }
    
    # Historique par consommateur
    consumer_nodes = real_env.consumer_nodes
    nodes_hist = {nid: {"T_in": [], "m_in": [], "p_dem": [], "p_sup": []} for nid in consumer_nodes}
    
    # Historique des vannes (Splits)
    branching_nodes = real_env.branching_nodes
    split_hist = {} 

    # Reset
    obs = env.reset()
    
    # Pré-calcul des index de tuyaux entrants pour chaque nœud (pour sommer les débits rapidement)
    parents_map = real_env.graph.get_parent_nodes()
    incoming_pipes_cache = {}
    for nid in consumer_nodes:
        incoming_pipes_cache[nid] = []
        for p_node in parents_map.get(nid, []):
            edge = real_env.network.graph.edges[(p_node, nid)]
            incoming_pipes_cache[nid].append(edge["pipe_index"])

    print(" Simulation en cours...")
    while True:
        action, _ = model.predict(obs, deterministic=True)
        
        # --- ETAPE PRINCIPALE ---
        # On récupère 'infos' qui contient les calculs physiques faits dans step()
        obs, rewards, dones, infos = env.step(action)
        
        if dones[0]: 
            break
        
        # infos est une liste (car VecEnv), on prend le premier élément
        info = infos[0]
        
        # --- EXTRACTION DES DONNÉES (Via INFO) ---
        # On suppose que vous avez ajouté "pipe_mass_flows" et "node_temperatures" dans le dict info de step()
        try:
            pipe_flows = info["pipe_mass_flows"]
            node_temps = info["node_temperatures"]
        except KeyError:
            print("ERREUR: Le dictionnaire 'info' ne contient pas 'pipe_mass_flows' ou 'node_temperatures'.")
            print("Vérifiez que vous avez bien modifié district_heating_gym_env.py pour inclure ces données dans le return de step().")
            sys.exit(1)

        t = real_env.current_t
        
        # Stockage Variables Globales
        history["time"].append(t)
        history["T_inlet"].append(real_env.actual_inlet_temp)
        history["m_inlet"].append(real_env.actual_mass_flow)
        
        # Pour la demande/fourni total, on peut utiliser les attributs de l'env s'ils sont à jour,
        # ou les recalculer à partir des données extraites.
        history["demand_total"].append(real_env.last_total_p_demand)
        history["supplied_total"].append(real_env.last_total_p_supplied)

        # Stockage Variables Hydrauliques (Splits)
        # Les splits sont stockés dans le graphe, mis à jour par l'action courante
        for (u, v), edge_data in real_env.network.graph.edges.items():
            if u in branching_nodes:
                key = f"Split {u}->{v}"
                if key not in split_hist: split_hist[key] = []
                split_hist[key].append(edge_data.get("split_fraction", 0.0))

        # Stockage Variables Par Nœud (Consommateurs)
        for nid in consumer_nodes:
            # 1. Température (directement depuis le vecteur info)
            node_idx = real_env.graph.get_id_from_node(nid)
            T_val = node_temps[node_idx]
            
            # 2. Débit entrant (somme des vecteurs info)
            m_in_val = sum(pipe_flows[pidx] for pidx in incoming_pipes_cache[nid])
            
            # 3. Puissances (Calcul local rapide)
            p_dem = real_env.demand_funcs[nid](t)
            
            # Calcul du fourni borné par la physique (T_val vient de info)
            delta_T = max(T_val - config.SIMULATION_PARAMS.get("min_return_temp", 40.0), 0.0)
            p_sup = min(p_dem, m_in_val * real_env.props["cp"] * delta_T)
            
            nodes_hist[nid]["T_in"].append(T_val)
            nodes_hist[nid]["m_in"].append(m_in_val)
            nodes_hist[nid]["p_dem"].append(p_dem)
            nodes_hist[nid]["p_sup"].append(p_sup)
    
    print(" Simulation terminée. Génération du Tableau de Bord...")
    
    # --- VISUALISATION (DASHBOARD) ---
    t_h = np.array(history["time"]) / 3600.0
    
    # Style
    try: plt.style.use('seaborn-v0_8-whitegrid')
    except: pass
    
    # Figure avec 5 panneaux verticaux partageant l'axe X
    fig, axes = plt.subplots(5, 1, figsize=(14, 20), sharex=True)
    
    # Palette de couleurs pour distinguer les noeuds
    colors = plt.cm.tab10(np.linspace(0, 1, len(consumer_nodes)))
    
    # 1. ACTIONS SOURCE
    ax = axes[0]
    ax.plot(t_h, history["T_inlet"], color='#D62728', linewidth=2, label="T Source (°C)")
    ax_bis = ax.twinx()
    ax_bis.plot(t_h, history["m_inlet"], color='#1F77B4', linewidth=2, linestyle='--', label="Flow Source (kg/s)")
    ax.set_ylabel("Température (°C)", color='#D62728', fontweight='bold')
    ax_bis.set_ylabel("Débit Massique (kg/s)", color='#1F77B4', fontweight='bold')
    ax.set_title("1. Actions à la Source (Commandes Agent)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 2. PERFORMANCE GLOBALE
    ax = axes[1]
    dem_tot = np.array(history["demand_total"]) / 1e6 # MW
    sup_tot = np.array(history["supplied_total"]) / 1e6 # MW
    ax.fill_between(t_h, dem_tot, sup_tot, color='red', alpha=0.2, label="Déficit Global (Non fourni)")
    ax.plot(t_h, dem_tot, 'k--', linewidth=1.5, label="Total Demand")
    ax.plot(t_h, sup_tot, 'g-', linewidth=2, label="Total Supplied")
    ax.set_ylabel("Puissance (MW)")
    ax.set_title("2. Bilan de Puissance Global", fontsize=12, fontweight='bold')
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # 3. HYDRAULIQUE (SPLITS)
    ax = axes[2]
    for key, vals in split_hist.items():
        ax.plot(t_h, vals, label=key, linewidth=1.5)
    ax.set_ylabel("Ouverture Vanne [0-1]")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("3. Répartition Hydraulique (Positions Vannes)", fontsize=12, fontweight='bold')
    ax.legend(loc="lower center", ncol=min(len(split_hist), 4), fontsize='small', bbox_to_anchor=(0.5, -0.35))
    ax.grid(True, alpha=0.3)
    
    # 4. DÉBITS LOCAUX (Par Nœud)
    ax = axes[3]
    for i, nid in enumerate(consumer_nodes):
        ax.plot(t_h, nodes_hist[nid]["m_in"], color=colors[i], label=f"Node {nid}")
    ax.set_ylabel("Débit Local (kg/s)")
    ax.set_title("4. Distribution des Débits aux Consommateurs", fontsize=12, fontweight='bold')
    ax.legend(loc="upper right", fontsize='small', ncol=2)
    ax.grid(True, alpha=0.3)
    
    # 5. CONFORT THERMIQUE (Températures reçues)
    ax = axes[4]
    for i, nid in enumerate(consumer_nodes):
        ax.plot(t_h, nodes_hist[nid]["T_in"], color=colors[i], label=f"Node {nid}", linewidth=1.5)
    
    # Ligne T_min retour pour référence
    t_min_ref = config.SIMULATION_PARAMS.get("min_return_temp", 40.0)
    ax.axhline(t_min_ref, color='black', linestyle=':', linewidth=2, label="T_min (Retour)")
    
    ax.set_ylabel("Température Arrivée (°C)")
    ax.set_xlabel("Temps de simulation (heures)")
    ax.set_title("5. Températures reçues par les Consommateurs", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarde
    out_file = os.path.join(PLOTS_DIR, f"dashboard_FULL_{MODEL_SUBDIR}_{iteration}.png")
    plt.savefig(out_file, dpi=150)
    print(f" Graphique sauvegardé : {out_file}")
    plt.show()

if __name__ == "__main__":
    main()