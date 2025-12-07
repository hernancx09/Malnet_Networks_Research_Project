#!/usr/bin/env python3
"""
Load and process MalNet-Tiny dataset
MalNet-Tiny contains malware call graphs for classification
"""

import os
import json
import networkx as nx
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pickle

class MalNetTinyLoader:
    """
    Loader for MalNet-Tiny dataset
    """
    
    def __init__(self, data_dir: str = "data/malnet_tiny"):
        """
        Initialize MalNet-Tiny loader
        
        Args:
            data_dir: Directory containing MalNet-Tiny dataset
        """
        self.data_dir = Path(data_dir)
        self.graphs = []
        self.labels = []
        self.families = []
        
    def load_graphs(self, format: str = "edgelist", directed: bool = True) -> List[nx.DiGraph]:
        """
        Load graphs from MalNet-Tiny dataset
        
        Args:
            format: Graph format ('edgelist', 'pickle', 'json')
            directed: Load as directed graphs (default: True for call graphs)
        
        Returns:
            List of NetworkX graphs (DiGraph if directed=True, Graph otherwise)
        """
        graphs = []
        
        if not self.data_dir.exists():
            print(f"[ERROR] Dataset directory not found: {self.data_dir}")
            print("Please download MalNet-Tiny dataset first.")
            return graphs
        
        # Try different possible structures
        possible_paths = [
            self.data_dir / "graphs",
            self.data_dir / "data",
            self.data_dir,
        ]
        
        graph_dir = None
        for path in possible_paths:
            if path.exists():
                graph_dir = path
                break
        
        if graph_dir is None:
            print(f"[ERROR] Could not find graph files in {self.data_dir}")
            return graphs
        
        print(f"Loading graphs from: {graph_dir}")
        
        # Load graphs based on format
        graph_type = nx.DiGraph if directed else nx.Graph
        if format == "edgelist":
            graphs = self._load_edgelist(graph_dir, graph_type)
        elif format == "pickle":
            graphs = self._load_pickle(graph_dir)
        elif format == "json":
            graphs = self._load_json(graph_dir)
        
        self.graphs = graphs
        print(f"[SUCCESS] Loaded {len(graphs)} graphs")
        
        return graphs
    
    def _load_edgelist(self, graph_dir: Path, graph_type=nx.DiGraph) -> List:
        """Load graphs from edge list files (recursively)"""
        graphs = []
        labels = []
        
        # Look for .edgelist files recursively
        graph_files = list(graph_dir.rglob("*.edgelist"))
        
        if not graph_files:
            # Try direct .txt files
            graph_files = list(graph_dir.glob("*.txt"))
        
        print(f"Found {len(graph_files)} graph files")
        
        for graph_file in graph_files:
            try:
                G = nx.read_edgelist(str(graph_file), nodetype=int, create_using=graph_type, comments='#')
                graphs.append(G)
                
                # Extract label from path (family/subfamily)
                # e.g., malnet-graphs-tiny/adware/airpush/graph.edgelist
                parts = graph_file.parts
                if len(parts) >= 3:
                    # Get family and subfamily
                    family = parts[-3] if len(parts) >= 3 else "unknown"
                    subfamily = parts[-2] if len(parts) >= 2 else "unknown"
                    labels.append(f"{family}/{subfamily}")
                else:
                    labels.append("unknown")
                    
            except Exception as e:
                print(f"Warning: Could not load {graph_file}: {e}")
        
        # Store labels if we extracted them
        if labels and len(labels) == len(graphs):
            self.labels = labels
        
        return graphs
    
    def _load_pickle(self, graph_dir: Path) -> List[nx.Graph]:
        """Load graphs from pickle files"""
        graphs = []
        graph_files = sorted(graph_dir.glob("*.pkl")) + sorted(graph_dir.glob("*.pickle"))
        
        for graph_file in graph_files:
            try:
                with open(graph_file, 'rb') as f:
                    G = pickle.load(f)
                    if isinstance(G, nx.Graph):
                        graphs.append(G)
            except Exception as e:
                print(f"Warning: Could not load {graph_file}: {e}")
        
        return graphs
    
    def _load_json(self, graph_dir: Path) -> List[nx.Graph]:
        """Load graphs from JSON files"""
        graphs = []
        graph_files = sorted(graph_dir.glob("*.json"))
        
        for graph_file in graph_files:
            try:
                with open(graph_file, 'r') as f:
                    data = json.load(f)
                    G = nx.node_link_graph(data)
                    graphs.append(G)
            except Exception as e:
                print(f"Warning: Could not load {graph_file}: {e}")
        
        return graphs
    
    def load_labels(self, label_file: Optional[str] = None) -> List[str]:
        """
        Load graph labels (malware families)
        
        Args:
            label_file: Path to label file (if None, searches for common names)
        
        Returns:
            List of family labels
        """
        if label_file is None:
            # Try common label file names
            possible_files = [
                self.data_dir / "labels.txt",
                self.data_dir / "families.txt",
                self.data_dir / "labels.json",
                self.data_dir / "metadata.json",
            ]
            
            for file_path in possible_files:
                if file_path.exists():
                    label_file = str(file_path)
                    break
        
        if label_file is None or not os.path.exists(label_file):
            print(f"[WARNING] Label file not found. Using default labels.")
            self.labels = [f"family_{i}" for i in range(len(self.graphs))]
            return self.labels
        
        # Load labels
        labels = []
        if label_file.endswith('.json'):
            with open(label_file, 'r') as f:
                data = json.load(f)
                labels = data.get('labels', [])
        else:
            with open(label_file, 'r') as f:
                labels = [line.strip() for line in f if line.strip()]
        
        self.labels = labels
        print(f"[SUCCESS] Loaded {len(labels)} labels")
        
        return labels
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the loaded dataset
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self.graphs:
            return {}
        
        num_nodes = [G.number_of_nodes() for G in self.graphs]
        num_edges = [G.number_of_edges() for G in self.graphs]
        densities = [nx.density(G) for G in self.graphs]
        
        stats = {
            'num_graphs': len(self.graphs),
            'total_nodes': sum(num_nodes),
            'total_edges': sum(num_edges),
            'avg_nodes': np.mean(num_nodes),
            'avg_edges': np.mean(num_edges),
            'median_nodes': np.median(num_nodes),
            'median_edges': np.median(num_edges),
            'avg_density': np.mean(densities),
            'median_density': np.median(densities),
            'min_nodes': min(num_nodes),
            'max_nodes': max(num_nodes),
        }
        
        if self.labels:
            from collections import Counter
            family_counts = Counter(self.labels)
            stats['num_families'] = len(family_counts)
            stats['family_distribution'] = dict(family_counts)
        
        return stats


def download_instructions():
    """Print instructions for downloading MalNet-Tiny"""
    print("="*60)
    print("MalNet-Tiny Dataset Download Instructions")
    print("="*60)
    print("\nMalNet-Tiny is a dataset of malware call graphs.")
    print("It contains ~5,000 graphs from 47 malware families.")
    print("\nDownload options:")
    print("\n1. Official Source:")
    print("   - Visit: https://github.com/nd7141/graph_datasets")
    print("   - Or search for 'MalNet-Tiny' on GitHub")
    print("\n2. Alternative:")
    print("   - Check: https://drive.google.com (search 'MalNet-Tiny')")
    print("   - Or contact the authors of the research paper")
    print("\n3. After downloading:")
    print("   - Extract to: project/data/malnet_tiny/")
    print("   - Expected structure:")
    print("     project/data/malnet_tiny/")
    print("       graphs/          (graph files)")
    print("       labels.txt       (family labels)")
    print("       metadata.json    (optional)")
    print("\n" + "="*60)


if __name__ == "__main__":
    import sys
    
    # Change to project directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    print("MalNet-Tiny Dataset Loader")
    print("="*60)
    
    # Check if dataset exists
    data_dir = project_root / "data" / "malnet_tiny"
    
    if not data_dir.exists():
        print(f"\n[INFO] Dataset directory not found: {data_dir}")
        download_instructions()
    else:
        print(f"\n[INFO] Found dataset directory: {data_dir}")
        
        # Try to load
        loader = MalNetTinyLoader(str(data_dir))
        graphs = loader.load_graphs()
        
        if graphs:
            stats = loader.get_statistics()
            print("\n" + "="*60)
            print("Dataset Statistics")
            print("="*60)
            for key, value in stats.items():
                if key != 'family_distribution':
                    print(f"  {key}: {value}")
            
            if 'family_distribution' in stats:
                print(f"\n  Family distribution (top 10):")
                sorted_families = sorted(stats['family_distribution'].items(), 
                                       key=lambda x: x[1], reverse=True)[:10]
                for family, count in sorted_families:
                    print(f"    {family}: {count} graphs")

