#!/usr/bin/env python3
"""
ORCA (Optimized Routing and Connectivity Analysis) Demo
A demonstration of network modeling and analysis capabilities
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
import json

# Try to import ORCA integration
try:
    from orca_integration import compute_gdvs_with_orca, compute_gcm, check_orca_available
    ORCA_AVAILABLE = True
except ImportError:
    ORCA_AVAILABLE = False
    print("[INFO] ORCA integration not available. Install ORCA for graphlet features.")

class ORCANetworkModel:
    """
    ORCA Network Model - Simulates and analyzes network topologies
    """
    
    def __init__(self, name: str = "Network Model"):
        self.name = name
        self.graph = nx.Graph()
        self.nodes_data = {}
        self.edges_data = {}
        
    def add_node(self, node_id: str, capacity: float = 1.0, 
                 position: Optional[Tuple[float, float]] = None):
        """Add a node to the network"""
        self.graph.add_node(node_id, capacity=capacity)
        if position:
            self.nodes_data[node_id] = {'position': position, 'capacity': capacity}
        else:
            self.nodes_data[node_id] = {'capacity': capacity}
    
    def add_edge(self, node1: str, node2: str, 
                 bandwidth: float = 1.0, latency: float = 1.0,
                 cost: float = 1.0):
        """Add an edge (link) between two nodes"""
        self.graph.add_edge(node1, node2, 
                           bandwidth=bandwidth, 
                           latency=latency,
                           cost=cost)
        self.edges_data[(node1, node2)] = {
            'bandwidth': bandwidth,
            'latency': latency,
            'cost': cost
        }
    
    def shortest_path(self, source: str, target: str, 
                     metric: str = 'latency') -> List[str]:
        """Find shortest path using specified metric"""
        if metric == 'latency':
            return nx.shortest_path(self.graph, source, target, weight='latency')
        elif metric == 'cost':
            return nx.shortest_path(self.graph, source, target, weight='cost')
        elif metric == 'hops':
            return nx.shortest_path(self.graph, source, target)
        else:
            return nx.shortest_path(self.graph, source, target, weight=metric)
    
    def analyze_connectivity(self) -> Dict:
        """Analyze network connectivity metrics"""
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_connected(self.graph),
            'average_clustering': nx.average_clustering(self.graph),
            'diameter': nx.diameter(self.graph) if nx.is_connected(self.graph) else None,
            'average_path_length': nx.average_shortest_path_length(self.graph) 
                if nx.is_connected(self.graph) else None
        }
    
    def calculate_centrality(self) -> Dict[str, Dict]:
        """Calculate various centrality measures"""
        return {
            'degree': nx.degree_centrality(self.graph),
            'betweenness': nx.betweenness_centrality(self.graph),
            'closeness': nx.closeness_centrality(self.graph),
            'eigenvector': nx.eigenvector_centrality(self.graph, max_iter=1000)
        }
    
    def visualize(self, title: str = None, highlight_path: List[str] = None,
                 save_path: Optional[str] = None):
        """Visualize the network topology"""
        plt.figure(figsize=(12, 8))
        
        # Use spring layout for positioning
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # Draw nodes
        node_colors = []
        if highlight_path:
            node_colors = ['red' if node in highlight_path else 'lightblue' 
                          for node in self.graph.nodes()]
        else:
            node_colors = 'lightblue'
        
        nx.draw_networkx_nodes(self.graph, pos, 
                              node_color=node_colors,
                              node_size=1000,
                              alpha=0.9)
        
        # Draw edges
        if highlight_path:
            path_edges = set()
            for i in range(len(highlight_path)-1):
                path_edges.add((highlight_path[i], highlight_path[i+1]))
                path_edges.add((highlight_path[i+1], highlight_path[i]))  # Both directions
            
            edge_colors = []
            edge_widths = []
            for edge in self.graph.edges():
                if edge in path_edges:
                    edge_colors.append('red')
                    edge_widths.append(3)
                else:
                    edge_colors.append('gray')
                    edge_widths.append(2)
        else:
            edge_colors = 'gray'
            edge_widths = 2
        
        nx.draw_networkx_edges(self.graph, pos,
                              edge_color=edge_colors,
                              width=edge_widths,
                              alpha=0.6)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos,
                               font_size=10,
                               font_weight='bold')
        
        # Draw edge labels (bandwidth)
        edge_labels = {(u, v): f"{self.graph[u][v]['bandwidth']:.1f}" 
                      for u, v in self.graph.edges()}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels,
                                    font_size=8)
        
        plt.title(title or f"ORCA Network Model: {self.name}", 
                 fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[SUCCESS] Visualization saved to {save_path}")
        else:
            plt.show()
    
    def compute_graphlet_features(self, max_graphlet_size: int = 4):
        """
        Compute Graphlet Degree Vectors using ORCA (if available)
        
        Args:
            max_graphlet_size: Maximum graphlet size (2, 3, or 4)
        
        Returns:
            GDV matrix or None if ORCA is not available
        """
        if not ORCA_AVAILABLE:
            return None
        try:
            return compute_gdvs_with_orca(self.graph, max_graphlet_size)
        except:
            return None
    
    def compute_gcm_features(self, max_graphlet_size: int = 4):
        """
        Compute Graphlet Correlation Matrix (GCM) features
        
        Args:
            max_graphlet_size: Maximum graphlet size (2, 3, or 4)
        
        Returns:
            GCM vector or None if ORCA is not available
        """
        gdvs = self.compute_graphlet_features(max_graphlet_size)
        if gdvs is None:
            return None
        return compute_gcm(gdvs)
    
    def print_analysis_report(self):
        """Print a comprehensive analysis report"""
        print("\n" + "="*60)
        print(f"ORCA NETWORK ANALYSIS REPORT: {self.name}")
        print("="*60)
        
        # Connectivity Analysis
        connectivity = self.analyze_connectivity()
        print("\n[CONNECTIVITY METRICS]")
        print(f"  Nodes: {connectivity['num_nodes']}")
        print(f"  Edges: {connectivity['num_edges']}")
        print(f"  Density: {connectivity['density']:.4f}")
        print(f"  Connected: {connectivity['is_connected']}")
        if connectivity['diameter']:
            print(f"  Diameter: {connectivity['diameter']}")
            print(f"  Average Path Length: {connectivity['average_path_length']:.4f}")
        print(f"  Average Clustering: {connectivity['average_clustering']:.4f}")
        
        # Graphlet Features (if ORCA is available)
        if ORCA_AVAILABLE and check_orca_available():
            print("\n[GRAPHLET FEATURES]")
            gdvs = self.compute_graphlet_features(max_graphlet_size=4)
            if gdvs is not None:
                print(f"  GDV matrix shape: {gdvs.shape} (nodes × orbits)")
                gcm = self.compute_gcm_features(max_graphlet_size=4)
                if gcm is not None:
                    print(f"  GCM vector length: {len(gcm)}")
            else:
                print("  ORCA computation not available")
        
        # Centrality Analysis
        centrality = self.calculate_centrality()
        print("\n[TOP NODES BY CENTRALITY]")
        
        print("\n  Degree Centrality:")
        sorted_degree = sorted(centrality['degree'].items(), 
                              key=lambda x: x[1], reverse=True)[:5]
        for node, value in sorted_degree:
            print(f"    {node}: {value:.4f}")
        
        print("\n  Betweenness Centrality:")
        sorted_between = sorted(centrality['betweenness'].items(), 
                               key=lambda x: x[1], reverse=True)[:5]
        for node, value in sorted_between:
            print(f"    {node}: {value:.4f}")
        
        print("\n  Closeness Centrality:")
        sorted_close = sorted(centrality['closeness'].items(), 
                             key=lambda x: x[1], reverse=True)[:5]
        for node, value in sorted_close:
            print(f"    {node}: {value:.4f}")
        
        print("="*60 + "\n")


def demo_star_topology():
    """Demo: Star Network Topology"""
    print("\n" + "="*60)
    print("DEMO 1: STAR NETWORK TOPOLOGY")
    print("="*60)
    
    network = ORCANetworkModel("Star Network")
    
    # Create star topology: central hub with 5 nodes
    hub = "Hub"
    network.add_node(hub, capacity=10.0)
    
    for i in range(1, 6):
        node = f"Node{i}"
        network.add_node(node, capacity=5.0)
        network.add_edge(hub, node, bandwidth=10.0, latency=1.0, cost=1.0)
    
    network.print_analysis_report()
    network.visualize(title="Star Network Topology", save_path="star_network.png")
    
    # Find path
    path = network.shortest_path("Node1", "Node3", metric='hops')
    print(f"\nPath from Node1 to Node3: {' -> '.join(path)}")
    network.visualize(title="Star Network - Path from Node1 to Node3", 
                     highlight_path=path, save_path="star_network_path.png")
    
    return network


def demo_mesh_topology():
    """Demo: Mesh Network Topology"""
    print("\n" + "="*60)
    print("DEMO 2: MESH NETWORK TOPOLOGY")
    print("="*60)
    
    network = ORCANetworkModel("Mesh Network")
    
    # Create 6-node mesh network
    nodes = [f"Router{i}" for i in range(1, 7)]
    for node in nodes:
        network.add_node(node, capacity=8.0)
    
    # Create mesh connections with varying properties
    connections = [
        ("Router1", "Router2", 10.0, 2.0, 1.0),
        ("Router1", "Router3", 8.0, 3.0, 2.0),
        ("Router2", "Router3", 12.0, 1.5, 1.0),
        ("Router2", "Router4", 9.0, 2.5, 2.0),
        ("Router3", "Router5", 11.0, 2.0, 1.5),
        ("Router4", "Router5", 10.0, 1.0, 1.0),
        ("Router4", "Router6", 8.0, 3.0, 2.5),
        ("Router5", "Router6", 12.0, 1.5, 1.0),
        ("Router1", "Router6", 7.0, 4.0, 3.0),  # Long-distance link
    ]
    
    for node1, node2, bw, lat, cost in connections:
        network.add_edge(node1, node2, bandwidth=bw, latency=lat, cost=cost)
    
    network.print_analysis_report()
    network.visualize(title="Mesh Network Topology", save_path="mesh_network.png")
    
    # Find optimal paths using different metrics
    print("\n[PATH ANALYSIS]")
    source, target = "Router1", "Router6"
    
    path_latency = network.shortest_path(source, target, metric='latency')
    path_cost = network.shortest_path(source, target, metric='cost')
    path_hops = network.shortest_path(source, target, metric='hops')
    
    print(f"\nPath from {source} to {target}:")
    print(f"  By Latency: {' -> '.join(path_latency)}")
    print(f"  By Cost: {' -> '.join(path_cost)}")
    print(f"  By Hops: {' -> '.join(path_hops)}")
    
    network.visualize(title=f"Mesh Network - Optimal Path ({source} to {target})",
                     highlight_path=path_latency, 
                     save_path="mesh_network_path.png")
    
    return network


def demo_ring_topology():
    """Demo: Ring Network Topology"""
    print("\n" + "="*60)
    print("DEMO 3: RING NETWORK TOPOLOGY")
    print("="*60)
    
    network = ORCANetworkModel("Ring Network")
    
    # Create ring topology with 8 nodes
    num_nodes = 8
    nodes = [f"Switch{i}" for i in range(1, num_nodes + 1)]
    
    for node in nodes:
        network.add_node(node, capacity=6.0)
    
    # Connect in a ring
    for i in range(num_nodes):
        network.add_edge(nodes[i], nodes[(i + 1) % num_nodes], 
                        bandwidth=10.0, latency=2.0, cost=1.0)
    
    # Add some cross-connections
    network.add_edge("Switch1", "Switch4", bandwidth=8.0, latency=1.5, cost=1.5)
    network.add_edge("Switch2", "Switch6", bandwidth=8.0, latency=1.5, cost=1.5)
    
    network.print_analysis_report()
    network.visualize(title="Ring Network Topology", save_path="ring_network.png")
    
    return network


def demo_complex_network():
    """Demo: Complex Network with Multiple Metrics"""
    print("\n" + "="*60)
    print("DEMO 4: COMPLEX NETWORK ANALYSIS")
    print("="*60)
    
    network = ORCANetworkModel("Enterprise Network")
    
    # Create a more complex network
    nodes = {
        "Core": (0, 0),
        "Router1": (-2, 2),
        "Router2": (2, 2),
        "Router3": (-2, -2),
        "Router4": (2, -2),
        "Switch1": (-1, 1),
        "Switch2": (1, 1),
        "Switch3": (-1, -1),
        "Switch4": (1, -1),
    }
    
    for node, pos in nodes.items():
        capacity = 20.0 if node == "Core" else 10.0
        network.add_node(node, capacity=capacity, position=pos)
    
    # Core connections
    edges = [
        ("Core", "Router1", 20.0, 1.0, 1.0),
        ("Core", "Router2", 20.0, 1.0, 1.0),
        ("Core", "Router3", 20.0, 1.0, 1.0),
        ("Core", "Router4", 20.0, 1.0, 1.0),
        ("Router1", "Switch1", 10.0, 2.0, 2.0),
        ("Router1", "Switch2", 10.0, 2.0, 2.0),
        ("Router2", "Switch2", 10.0, 2.0, 2.0),
        ("Router3", "Switch3", 10.0, 2.0, 2.0),
        ("Router4", "Switch4", 10.0, 2.0, 2.0),
        ("Router1", "Router2", 15.0, 1.5, 1.5),
        ("Router3", "Router4", 15.0, 1.5, 1.5),
    ]
    
    for node1, node2, bw, lat, cost in edges:
        network.add_edge(node1, node2, bandwidth=bw, latency=lat, cost=cost)
    
    network.print_analysis_report()
    network.visualize(title="Enterprise Network Topology", 
                     save_path="enterprise_network.png")
    
    # Multi-path analysis
    print("\n[MULTI-PATH ANALYSIS]")
    source, target = "Switch1", "Switch4"
    
    # Find all simple paths
    all_paths = list(nx.all_simple_paths(network.graph, source, target))
    print(f"\nAll paths from {source} to {target}:")
    for i, path in enumerate(all_paths[:5], 1):  # Show first 5 paths
        total_latency = sum(network.graph[path[j]][path[j+1]]['latency'] 
                           for j in range(len(path)-1))
        print(f"  Path {i}: {' -> '.join(path)} (Total Latency: {total_latency})")
    
    optimal_path = network.shortest_path(source, target, metric='latency')
    network.visualize(title=f"Enterprise Network - Optimal Path ({source} to {target})",
                     highlight_path=optimal_path,
                     save_path="enterprise_network_path.png")
    
    return network


def main():
    """Run all ORCA demos"""
    print("\n" + "="*60)
    print("ORCA (Optimized Routing and Connectivity Analysis)")
    print("Network Modeling and Analysis Demo")
    print("="*60)
    
    # Run demos
    demo_star_topology()
    demo_mesh_topology()
    demo_ring_topology()
    demo_complex_network()
    
    print("\n" + "="*60)
    print("[SUCCESS] All demos completed successfully!")
    print("[SUCCESS] Network visualizations saved as PNG files")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

