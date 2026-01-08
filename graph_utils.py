from collections import deque

class Graph:
    """
    Analyse la structure du graphe et fournit des accesseurs optimisés pour NumPy (indices entiers).
    """
    def __init__(self, edges):
        if isinstance(edges, list):
            self.edges = edges
        else:
            self.edges = list(edges.keys())
            
        self._build_nodes()
        self._find_node_roles()
        self._compute_topological_order()

        # Structures optimisées pour la vectorisation
        self.int_adjacency = [[] for _ in range(self.n_nodes)]
        self.int_parent_adjacency = [[] for _ in range(self.n_nodes)]

    def _build_nodes(self):
        self.nodes = set()
        self.parent_nodes = {} 
        self.child_nodes = {}  
        self.out_degree = {}
        self.in_degree = {}
        
        for u, v in self.edges:
            self.nodes.add(u)
            self.nodes.add(v)
            self.child_nodes.setdefault(u, []).append(v)
            self.parent_nodes.setdefault(v, []).append(u)
            self.out_degree[u] = self.out_degree.get(u, 0) + 1
            self.in_degree[v] = self.in_degree.get(v, 0) + 1
            if u not in self.in_degree: self.in_degree[u] = 0
            if v not in self.out_degree: self.out_degree[v] = 0

        for n in self.child_nodes: self.child_nodes[n].sort()
        for n in self.parent_nodes: self.parent_nodes[n].sort()

        self.nodes_list = sorted(list(self.nodes))
        self.n_nodes = len(self.nodes_list)
        
        self.node_to_id = {n: i for i, n in enumerate(self.nodes_list)}
        self.id_to_node = {i: n for i, n in enumerate(self.nodes_list)}
        
        self.terminal_nodes = [n for n in self.nodes_list if not self.child_nodes.get(n)]

    def _find_node_roles(self):
        inlets = [n for n in self.nodes_list if n not in self.parent_nodes]
        self.inlet_node = inlets[0] if len(inlets) == 1 else inlets
        
        self.branching_nodes = sorted([n for n in self.nodes_list if self.out_degree.get(n, 0) > 1])
        self.consumer_nodes = sorted([n for n in self.nodes_list if n != self.inlet_node])
        
        self.branching_indices = [self.node_to_id[n] for n in self.branching_nodes]
        self.consumer_indices_list = [self.node_to_id[n] for n in self.consumer_nodes]

        self.downstream_consumers_indices_map = {}

        for b_node in self.branching_nodes:
            b_idx = self.node_to_id[b_node]
            self.downstream_consumers_indices_map[b_idx] = {}
            
            for child in self.child_nodes.get(b_node, []):
                child_idx = self.node_to_id[child]
                consumers_idx = []
                queue = deque([child])
                visited = {child}
                while queue:
                    curr = queue.popleft()
                    if curr in self.consumer_nodes:
                        consumers_idx.append(self.node_to_id[curr])
                    for gc in self.child_nodes.get(curr, []):
                        if gc not in visited:
                            visited.add(gc)
                            queue.append(gc)
                            
                self.downstream_consumers_indices_map[b_idx][child_idx] = consumers_idx

    def _compute_topological_order(self):
        in_degree = self.in_degree.copy()
        queue = [n for n in self.nodes_list if in_degree[n] == 0]
        self.topo_indices = []
        while queue:
            u = queue.pop(0)
            self.topo_indices.append(self.node_to_id[u])
            for v in self.child_nodes.get(u, []):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

    def get_id(self, node): return self.node_to_id[node]
    def get_node(self, idx): return self.id_to_node[idx]
    
    def register_pipe_index(self, u, v, pipe_idx):
        u_id, v_id = self.node_to_id[u], self.node_to_id[v]
        self.int_adjacency[u_id].append((v_id, pipe_idx))
        self.int_parent_adjacency[v_id].append((u_id, pipe_idx))

    def get_downstream_consumers_indices(self, branch_idx, child_idx):
        return self.downstream_consumers_indices_map.get(branch_idx, {}).get(child_idx, [])