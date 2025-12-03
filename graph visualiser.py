import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.cm import get_cmap

G = nx.Graph()

G.add_edge("A", "B", value=0.1)
G.add_edge("B", "C", value=0.7)
G.add_edge("A", "C", value=0.4)

pos = nx.spring_layout(G, seed=0)

fig, ax = plt.subplots()

# Dessiner les noeuds + labels
nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=800, ax=ax)
nx.draw_networkx_labels(G, pos, ax=ax)

cmap = get_cmap("viridis")

for u, v, data in G.edges(data=True):
    value = data["value"]  # valeur pour l'arête
    color = cmap(value)

    # coords des nœuds
    x1, y1 = pos[u]
    x2, y2 = pos[v]

    # segments (ici juste un segment unique → pas un vrai gradient)
    # Pour un gradient vrai, on subdivise :
    n = 20
    xs = np.linspace(x1, x2, n)
    ys = np.linspace(y1, y2, n)

    segments = [
        [[xs[i], ys[i]], [xs[i+1], ys[i+1]]]
        for i in range(n - 1)
    ]

    # gradient : couleur varie selon i
    colors = cmap(np.linspace(0, 1, n-1))

    lc = LineCollection(segments, colors=colors, linewidth=3)
    ax.add_collection(lc)

ax.set_aspect("equal")
plt.axis("off")
plt.show()
