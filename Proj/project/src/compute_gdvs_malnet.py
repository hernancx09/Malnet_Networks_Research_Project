#!/usr/bin/env python3
"""
Compute Graphlet Degree Vectors (GDVs) for MalNet-Tiny dataset using ORCA
As described in the research paper: "Classifying Malware Families Using 
Graphlet-Based Topological Signatures from MalNet-Tiny"
"""

import os
import sys
import numpy as np
import networkx as nx
from pathlib import Path
from typing import List, Dict, Optional
import pickle
import json

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from load_malnet import MalNetTinyLoader
from orca_integration import compute_gdvs_with_orca, compute_gcm

class GDVProcessor:
    """
    Process MalNet-Tiny graphs to compute GDVs and GCMs
    """
    
    def __init__(self, data_dir: str = "data/malnet_tiny", 
                 output_dir: str = "data/gdvs"):
        """
        Initialize GDV processor
        
        Args:
            data_dir: Directory containing MalNet-Tiny dataset
            output_dir: Directory to save computed GDVs and GCMs
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.gdvs = []
        self.gcms = []
        
    def process_graphs(self, graphs: List[nx.Graph], 
                      max_graphlet_size: int = 4,
                      save_individual: bool = False) -> Dict:
        """
        Process all graphs to compute GDVs and GCMs
        
        Args:
            graphs: List of NetworkX graphs
            max_graphlet_size: Maximum graphlet size (2, 3, or 4)
            save_individual: Whether to save GDVs for each graph separately
        
        Returns:
            Dictionary with processing results
        """
        print("="*60)
        print("Computing GDVs for MalNet-Tiny Graphs")
        print("="*60)
        print(f"Number of graphs: {len(graphs)}")
        print(f"Max graphlet size: {max_graphlet_size}")
        print()
        
        results = {
            'successful': 0,
            'failed': 0,
            'gdv_shapes': [],
            'gcm_lengths': [],
        }
        
        for i, graph in enumerate(graphs):
            if (i + 1) % 100 == 0:
                print(f"Processing graph {i+1}/{len(graphs)}...")
            
            try:
                # Compute GDV
                gdv = compute_gdvs_with_orca(graph, max_graphlet_size=max_graphlet_size)
                
                if gdv is not None:
                    self.gdvs.append(gdv)
                    results['gdv_shapes'].append(gdv.shape)
                    
                    # Compute GCM
                    gcm = compute_gcm(gdv)
                    if gcm is not None:
                        self.gcms.append(gcm)
                        results['gcm_lengths'].append(len(gcm))
                    
                    # Save individual if requested
                    if save_individual:
                        self._save_individual(i, gdv, gcm)
                    
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                    self.gdvs.append(None)
                    self.gcms.append(None)
                    
            except Exception as e:
                print(f"Error processing graph {i}: {e}")
                results['failed'] += 1
                self.gdvs.append(None)
                self.gcms.append(None)
        
        # Save all results
        self._save_all_results()
        
        print("\n" + "="*60)
        print("Processing Complete")
        print("="*60)
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        if results['gdv_shapes']:
            print(f"Average GDV shape: {np.mean([s[0] for s in results['gdv_shapes']])} nodes")
            print(f"Average orbits: {np.mean([s[1] for s in results['gdv_shapes']])}")
        print("="*60)
        
        return results
    
    def _save_individual(self, graph_id: int, gdv: np.ndarray, gcm: Optional[np.ndarray]):
        """Save individual graph's GDV and GCM"""
        gdv_file = self.output_dir / f"gdv_{graph_id}.npy"
        np.save(gdv_file, gdv)
        
        if gcm is not None:
            gcm_file = self.output_dir / f"gcm_{graph_id}.npy"
            np.save(gcm_file, gcm)
    
    def _save_all_results(self):
        """Save all GDVs and GCMs"""
        # Save GDVs
        gdvs_file = self.output_dir / "all_gdvs.pkl"
        with open(gdvs_file, 'wb') as f:
            pickle.dump(self.gdvs, f)
        print(f"\nSaved GDVs to: {gdvs_file}")
        
        # Save GCMs
        if self.gcms:
            gcms_file = self.output_dir / "all_gcms.pkl"
            with open(gcms_file, 'wb') as f:
                pickle.dump(self.gcms, f)
            print(f"Saved GCMs to: {gcms_file}")
        
        # Save metadata
        metadata = {
            'num_graphs': len(self.gdvs),
            'gdv_shapes': [g.shape if g is not None else None for g in self.gdvs],
            'gcm_lengths': [len(g) if g is not None else None for g in self.gcms],
        }
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to: {metadata_file}")


def main():
    """Main processing pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute GDVs for MalNet-Tiny')
    parser.add_argument('--data-dir', type=str, default='data/malnet_tiny',
                       help='Directory containing MalNet-Tiny dataset')
    parser.add_argument('--output-dir', type=str, default='data/gdvs',
                       help='Directory to save computed GDVs')
    parser.add_argument('--max-size', type=int, default=4,
                       help='Maximum graphlet size (2, 3, or 4)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of graphs to process (for testing)')
    
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Load dataset
    print("Loading MalNet-Tiny dataset...")
    loader = MalNetTinyLoader(args.data_dir)
    graphs = loader.load_graphs()
    
    if not graphs:
        print("[ERROR] No graphs loaded. Please check dataset directory.")
        return
    
    # Limit if specified
    if args.limit:
        graphs = graphs[:args.limit]
        print(f"Limited to {len(graphs)} graphs for processing")
    
    # Process
    processor = GDVProcessor(args.data_dir, args.output_dir)
    results = processor.process_graphs(graphs, max_graphlet_size=args.max_size)
    
    print("\n[SUCCESS] GDV computation complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

