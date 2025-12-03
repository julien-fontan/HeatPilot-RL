import numpy as np
import matplotlib.pyplot as plt

def _compute_node_positions(network):
    """
    Calcule des positions 2D des noeuds à partir de la topologie du réseau.

    Idée:
    - On définit un "niveau" (depth) pour chaque noeud à partir du noeud d'entrée.
    - Tous les noeuds d'un même niveau ont la même coordonnée x.
    - À l'intérieur d'un niveau, on répartit les noeuds verticalement (y) pour voir les branches.
    """
    # 1) On travaille en indices internes 0..n_nodes-1
    n_nodes = network.n_nodes
    u_indices = network.pipe_u_indices
    v_indices = network.pipe_v_indices

    # Déterminer le noeud source en index interne
    inlet_logical = network.inlet_node_id
    inlet_idx = network.node_map[inlet_logical]

    # BFS pour obtenir la profondeur (niveau) de chaque noeud
    depth = np.full(n_nodes, np.inf)
    depth[inlet_idx] = 0
    queue = [inlet_idx]
    while queue:
        u = queue.pop(0)
        for pipe_idx in network.node_out_pipes[u]:
            v = v_indices[pipe_idx]
            if depth[v] > depth[u] + 1:
                depth[v] = depth[u] + 1
                queue.append(v)

    # Certains noeuds peuvent rester à +inf (isolés) : on les met à la profondeur max+1
    finite_depths = depth[np.isfinite(depth)]
    if finite_depths.size == 0:
        finite_depths = np.array([0.0])
    max_depth = int(finite_depths.max())
    depth[~np.isfinite(depth)] = max_depth + 1

    # 2) Regrouper les noeuds par niveau
    level_to_nodes = {}
    for idx in range(n_nodes):
        d = int(depth[idx])
        level_to_nodes.setdefault(d, []).append(idx)

    # 3) Construire les coordonnées (x,y)
    positions = {}
    # On tri les niveaux croissants pour que le flux aille de gauche à droite
    for level in sorted(level_to_nodes.keys()):
        nodes_at_level = level_to_nodes[level]
        k = len(nodes_at_level)
        x = float(level)  # même x pour tout le niveau

        if k == 1:
            # un seul noeud : placer à y=0
            y_coords = [0.0]
        else:
            # plusieurs noeuds : les répartir symétriquement autour de 0
            # par exemple: [-1, 0, 1] ou [-1.5, -0.5, 0.5, 1.5], etc.
            y_coords = np.linspace(- (k - 1) / 2.0, (k - 1) / 2.0, k)

        for idx_node, y in zip(nodes_at_level, y_coords):
            logical_id = network.inv_node_map[idx_node]
            positions[logical_id] = (x, float(y))

    return positions

def plot_temperature_graph(network, state_vector, t, cmap="plasma"):
    """
    Visualise le graphe:
    - couleur des arrêtes: température moyenne dans la conduite
    - texte à côté de chaque arrête: mass flow
    - laisse de la place autour de chaque noeud pour futurs champs de texte
    """
    # Récupère profils de T et mass flows
    _, pipe_temperatures, mass_flows = network.get_pipe_temperatures_and_flows(t, state_vector)

    positions = _compute_node_positions(network)

    fig, ax = plt.subplots(figsize=(10, 4))

    # Échelle de couleurs sur les T moyennes
    mean_temps = np.array([T.mean() for T in pipe_temperatures])
    norm = plt.Normalize(vmin=mean_temps.min(), vmax=mean_temps.max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    # Tracer les arrêtes
    for i, p in enumerate(network.pipes):
        u_logical, v_logical = p.nodes
        x_u, y_u = positions[u_logical]
        x_v, y_v = positions[v_logical]

        color = sm.to_rgba(mean_temps[i])

        ax.plot([x_u, x_v], [y_u, y_v], color=color, linewidth=3)

        # Texte mass flow au milieu du segment
        mx = 0.5 * (x_u + x_v)
        my = 0.5 * (y_u + y_v) + 0.15
        ax.text(mx, my, f"{mass_flows[i]:.2f}", fontsize=8, ha="center", va="bottom")

    # Tracer les noeuds
    for node_id, (x, y) in positions.items():
        ax.scatter([x], [y], s=80, color="black", zorder=3)
        # Laisser de la place autour du noeud pour texte (future UI)
        ax.text(x, y - 0.3, f"{node_id}", ha="center", va="top", fontsize=8)

    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Température moyenne [°C]")

    ax.set_axis_off()
    fig.tight_layout()
    return fig, ax
