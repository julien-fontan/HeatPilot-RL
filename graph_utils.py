class Graph:
    """
    Analyse la structure du graphe et fournit des accesseurs pour les propriétés topologiques.
    """
    def __init__(self, edges):
        
        # Si edges est une liste de tuples, on convertit en dictionnaire pour permettre le stockage d'attributs
        if isinstance(edges, list):
            self.edges = {e: {} for e in edges}
        else:
            self.edges = edges
            
        self.n_edges = len(self.edges)
        
        self._build_nodes()
        self._find_node_roles()

        # Mappings vers des indices
        self.node_ids = {n: i for i, n in enumerate(self.nodes)}
        self.inv_node_ids = {i: n for i, n in enumerate(self.nodes)}
        
        self._compute_topological_order()   # Tri topologique (faire les calculs dans le bon ordre)

    def _build_nodes(self):
        self.nodes = set()
        self.parent_nodes = {}
        self.child_nodes = {}
        self.out_degree = {} # child count
        self.in_degree = {} # parent count
        
        for u, v in self.edges:
            self.nodes.add(u)
            self.nodes.add(v)
            if u not in self.child_nodes: self.child_nodes[u] = []
            self.child_nodes[u].append(v)
            if v not in self.parent_nodes: self.parent_nodes[v] = []
            self.parent_nodes[v].append(u)

            self.out_degree[u] = self.out_degree.get(u, 0) + 1
            if v not in self.out_degree: self.out_degree[v] = 0
            self.in_degree[v] = self.in_degree.get(v, 0) + 1
            if u not in self.in_degree: self.in_degree[u] = 0

        for i in self.child_nodes:  # on trie pour assurer le déterminisme
            self.child_nodes[i].sort()
        for i in self.parent_nodes:
            self.parent_nodes[i].sort()

        self.nodes = sorted(list(self.nodes))
        self.n_nodes = len(self.nodes)
        self.terminal_nodes = [n for n in self.nodes if n not in self.child_nodes or not self.child_nodes[n]]

    def _find_node_roles(self):
        inlets = [n for n in self.nodes if n not in self.parent_nodes]  # pas de parents
        if len(inlets) == 1:
            self.inlet_node = inlets[0]
        else:
            self.inlet_node = inlets
        self.branching_nodes = sorted([n for n in self.nodes if self.out_degree.get(n, 0) > 1])
        self.consumer_nodes = sorted([n for n in self.nodes if n != self.inlet_node])

    def _compute_topological_order(self):
        in_degree = self.in_degree.copy()
        queue = [n for n in self.nodes if in_degree[n] == 0]
        self.topo_nodes = []
        while queue:
            u = queue.pop(0)
            self.topo_nodes.append(self.node_ids[u])
            for v in self.child_nodes.get(u, []):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

    # --- Getters ---
    def get_id_from_node(self, node):
        return self.node_ids[node]

    def get_node_from_id(self, idx):
        return self.inv_node_ids[idx]

    def get_nodes_count(self):
        return self.n_nodes

    def get_parent_nodes(self):
        return self.parent_nodes
    
    def get_child_nodes(self):
        return self.child_nodes

    def get_inlet_node(self):
        return self.inlet_node
    
    def get_consumer_nodes(self):
        return self.consumer_nodes
    
    def get_branching_nodes(self):
        return self.branching_nodes

    def get_topo_nodes(self):
        return self.topo_nodes
    
    def get_terminal_nodes(self):
        return self.terminal_nodes