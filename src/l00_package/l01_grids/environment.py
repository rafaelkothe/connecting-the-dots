import numpy as np
import networkx as nx
import pandas as pd
from sklearn.preprocessing import normalize

class Grids:
    def __init__(self, simulation_params, random_seed):
        """
        Initialize the Grids class with simulation parameters and a random seed.

        Args:
            simulation_params (Params): Parameters for the simulation.
            random_seed (int): Random seed for reproducibility.
        """
        self.targeted_agent_centrality = simulation_params.tgt_agt_centr
        self._create_network(simulation_params, random_seed)
        self._initialize_matrices(simulation_params)

    def _create_network(self, simulation_params, random_seed):
        """
        Create the network graph and adjacency matrix.

        Args:
            simulation_params (Params): Parameters for the simulation.
            random_seed (int): Random seed for reproducibility.
        """
        num_agents = simulation_params.agent_number
        num_edges_to_attach = simulation_params.topology_m

        # Number of nodes
        num_nodes = num_agents

        # Target number of edges for all networks
        target_edge_count = 1200  # Adjust this to your desired edge count

        def generate_connected_scale_free_network(num_nodes, target_edge_count):
            while True:
                scale_free_network = nx.barabasi_albert_graph(num_nodes, 15, seed=random_seed)
                if nx.is_connected(scale_free_network):
                    return scale_free_network

        def generate_connected_random_network(num_nodes, target_edge_count):
            while True:
                random_network = nx.gnm_random_graph(num_nodes, target_edge_count, seed=random_seed)
                if nx.is_connected(random_network):
                    return random_network

        def generate_connected_small_world_network(num_nodes, target_edge_count):
            while True:
                p = target_edge_count / (num_nodes * (num_nodes - 1) / 2)
                small_world_network = nx.newman_watts_strogatz_graph(num_nodes, 25, p, seed=random_seed)
                if nx.is_connected(small_world_network):
                    return small_world_network

        def generate_connected_regular_network(num_nodes, target_edge_count):
            while True:
                degree = int(target_edge_count * 5 / num_nodes)  # Ensure an even number of edges for regularity
                regular_network = nx.random_regular_graph(degree, num_nodes, seed=random_seed)
                if nx.is_connected(regular_network):
                    return regular_network

        # Map network types to generator functions
        network_generators = {
            "albert-barabasi": generate_connected_scale_free_network,
            "random-graph": generate_connected_random_network,
            "watts-strogatz": generate_connected_small_world_network,
            "regular-graph": generate_connected_regular_network
        }

        network_graph = network_generators[simulation_params.topology](num_nodes, target_edge_count)

        # Network adjacency matrix
        self.adj_trust_matrix = nx.to_numpy_array(network_graph)

        # Normalize network weights
        self.adj_trust_matrix = normalize(self.adj_trust_matrix, axis=1, norm='l1')

        # Create agent network from the adjacency matrix
        self.agent_network = nx.from_numpy_array(self.adj_trust_matrix)

    def _initialize_matrices(self, simulation_params):
        """
        Initialize matrices A, B, C, b, and c used in the simulation.

        Args:
            simulation_params (Params): Parameters for the simulation.
        """
        a1, a2, b1, b2, c1, c2, c3 = (
            simulation_params.a1,
            simulation_params.a2,
            simulation_params.b1,
            simulation_params.b2,
            simulation_params.c1,
            simulation_params.c2,
            simulation_params.c3,
        )

        self.A = np.array([[1 + a2 * c2 * (1 - c3), a2 * c1 * (1 - c3)],
                           [-b2, 1]])

        self.B = np.array([[1 - a1, 0],
                           [0, 1 - b1]])

        self.C = np.array([[a1, a2],
                           [0, b1]])

        self.b = np.array([-a2 * c3, 0])
        self.c = np.array([-a2 * c1 * (c3 - 1), 0])

    def degree_centrality(self, G):
        """
        Calculate degree centrality for nodes in the agent network.

        Returns:
            pd.DataFrame: Sorted degree centrality values for nodes.
        """
        cc = nx.degree_centrality(G)
        centrality_df = pd.DataFrame.from_dict({
            'node': list(cc.keys()),
            'degree_centrality': list(cc.values())
        })
        return centrality_df.sort_values('degree_centrality', ascending=False).iloc[self.targeted_agent_centrality]

    def closeness_centrality(self):
        """
        Calculate closeness centrality for nodes in the agent network.

        Returns:
            pd.DataFrame: Sorted closeness centrality values for nodes.
        """
        cc = nx.closeness_centrality(self.agent_network)
        centrality_df = pd.DataFrame.from_dict({
            'node': list(cc.keys()),
            'closeness_centrality': list(cc.values())
        })
        return centrality_df.sort_values('closeness_centrality', ascending=False).iloc[self.targeted_agent_centrality]

    def betweenness_centrality(self):
        """
        Calculate betweenness centrality for nodes in the agent network.

        Returns:
            pd.DataFrame: Sorted betweenness centrality values for nodes.
        """
        cc = nx.betweenness_centrality(self.agent_network)
        centrality_df = pd.DataFrame.from_dict({
            'node': list(cc.keys()),
            'centrality': list(cc.values())
        })
        return centrality_df.sort_values('centrality', ascending=False).iloc[self.targeted_agent_centrality]

    def clustering(self, G):
        """
        Calculate clustering coefficient for nodes in the agent network.

        Returns:
            pd.DataFrame: Sorted clustering coefficient values for nodes.
        """
        cc = nx.clustering(G)
        clustering_df = pd.DataFrame.from_dict({
            'node': list(cc.keys()),
            'clustering': list(cc.values())
        })
        return clustering_df.sort_values('clustering', ascending=False).iloc[self.targeted_agent_centrality]

    def average_clustering(self, G):
        """
        Calculate the average clustering coefficient for the agent network.

        Returns:
            float: Average clustering coefficient.
        """
        return nx.average_clustering(G)
