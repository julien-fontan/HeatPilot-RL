import numpy as np

class Graph:
    """
    Analyse la structure du graphe et fournit des accesseurs pour les propriétés topologiques.
    """
    def __init__(self, edges):
        self.edges = edges
        self.n_pipes = len(edges)
        self.edge_index_map = {e: i for i, e in enumerate(edges)}
        
        # 1. Construction de base
        self.nodes = set()
        self.adj = {}
        self.in_degree = {}
        
        for u, v in self.edges:
            self.nodes.add(u)
            self.nodes.add(v)
            if u not in self.adj: self.adj[u] = []
            self.adj[u].append(v)
            
            self.in_degree[v] = self.in_degree.get(v, 0) + 1
            if u not in self.in_degree: self.in_degree[u] = 0

        self.unique_nodes = sorted(list(self.nodes))
        self.n_nodes = len(self.unique_nodes)
        
        # 2. Mappings
        self.node_map = {n: i for i, n in enumerate(self.unique_nodes)}
        self.inv_node_map = {i: n for i, n in enumerate(self.unique_nodes)}
        
        # 3. Analyse des rôles
        self._analyze_roles()
        
        # 4. Structures d'adjacence (Indices)
        self._build_adjacency_structures()
        
        # 5. Tri topologique
        self._compute_topological_order()

    def _analyze_roles(self):
        self.branching_map = {u: sorted(vs) for u, vs in self.adj.items() if len(vs) > 1}
        self.branching_nodes = sorted(list(self.branching_map.keys()))
        
        inlets = [n for n in self.unique_nodes if self.in_degree[n] == 0]
        self.inlet_node = inlets[0] if inlets else (min(self.unique_nodes) if self.unique_nodes else None)
        self.consumer_nodes = sorted([n for n in self.unique_nodes if n != self.inlet_node])

    def _build_adjacency_structures(self):
        self.node_in_pipes = [[] for _ in range(self.n_nodes)]
        self.node_out_pipes = [[] for _ in range(self.n_nodes)]
        self.pipe_u_indices = np.zeros(self.n_pipes, dtype=int)
        self.pipe_v_indices = np.zeros(self.n_pipes, dtype=int)

        for i, (u, v) in enumerate(self.edges):
            u_idx = self.node_map[u]
            v_idx = self.node_map[v]
            
            self.pipe_u_indices[i] = u_idx
            self.pipe_v_indices[i] = v_idx
            
            self.node_out_pipes[u_idx].append(i)
            self.node_in_pipes[v_idx].append(i)

    def _compute_topological_order(self):
        indeg = np.array([len(self.node_in_pipes[v]) for v in range(self.n_nodes)], dtype=int)
        queue = [i for i in range(self.n_nodes) if indeg[i] == 0]
        topo = []
        while queue:
            n = queue.pop(0)
            topo.append(n)
            for pipe_idx in self.node_out_pipes[n]:
                v = self.pipe_v_indices[pipe_idx]
                indeg[v] -= 1
                if indeg[v] == 0:
                    queue.append(v)
        self.topo_nodes = np.array(topo, dtype=int)

    # --- Getters ---
    def get_node_maps(self):
        return self.node_map, self.inv_node_map

    def get_nodes_count(self):
        return self.n_nodes

    def get_adjacency_indices(self):
        return self.node_in_pipes, self.node_out_pipes, self.pipe_u_indices, self.pipe_v_indices

    def get_topological_sort(self):
        return self.topo_nodes

    def get_special_nodes(self):
        return self.inlet_node, self.consumer_nodes, self.branching_nodes

    def get_branching_map(self):
        return self.branching_map
