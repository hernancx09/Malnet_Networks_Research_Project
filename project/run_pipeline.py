#!/usr/bin/env python3
"""
Main pipeline for computing Directed Graphlet Degree Vectors (DGDVs)
for MalNet-Tiny dataset using UCL Directed Graphlet Counter
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from load_malnet import MalNetTinyLoader
from compute_dgdvs_malnet import DGDVProcessor
import numpy as np
import networkx as nx

def main():
    """Main pipeline execution"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compute DGDVs for MalNet-Tiny dataset')
    parser.add_argument('-test', '--test', action='store_true',
                       help='Run test mode: use test edgelist and compute 2-4 node graphlets on 1 graph')
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("="*60)
    if args.test:
        print("MalNet-Tiny DGDV Computation Pipeline - TEST MODE")
    else:
        print("MalNet-Tiny DGDV Computation Pipeline")
    print("="*60)
    
    if args.test:
        # Test mode: use test edgelist
        test_mode(project_root)
    else:
        # Normal mode: use full dataset
        normal_mode(project_root)


def test_mode(project_root: Path):
    """Run pipeline in test mode with test edgelist"""
    print("\n[TEST MODE] Using test edgelist from test/data/")
    
    # Find test edgelist files
    test_data_dir = project_root.parent / "test" / "data"
    test_edgelists = list(test_data_dir.glob("*.edgelist"))
    
    if not test_edgelists:
        print("[ERROR] No test edgelist files found in test/data/")
        print("  Please ensure test edgelist files exist")
        return
    
    # Use the first available test edgelist
    test_edgelist = test_edgelists[0]
    print(f"  Using test graph: {test_edgelist.name}")
    
    # Load the test graph
    try:
        graph = nx.read_edgelist(str(test_edgelist), nodetype=int, create_using=nx.DiGraph, comments='#')
        print(f"  Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    except Exception as e:
        print(f"[ERROR] Failed to load test graph: {e}")
        return
    
    # Configuration for test mode
    output_dir = "data/gdvs_test"
    min_graphlet_size = 2  # Test 2-4 node graphlets
    max_graphlet_size = 4
    
    print(f"\n[1/2] Computing DGDVs (2-4 node graphlets)...")
    print(f"  Graphlet size range: {min_graphlet_size}-{max_graphlet_size} node")
    
    # Use DGDVProcessor for consistency
    processor = DGDVProcessor("data/malnet_tiny", output_dir, use_gpu=False)  # Disable GPU for quick test
    
    # Process single graph
    results = processor.process_graphs(
        [graph],
        labels=["test"],
        min_graphlet_size=min_graphlet_size,
        max_graphlet_size=max_graphlet_size
    )
    
    # Summary
    print(f"\n[2/2] Test complete!")
    print(f"  Successful: {results['metadata']['successful']}")
    print(f"  Failed: {results['metadata']['failed']}")
    if results['dgdvs']:
        dgdv = results['dgdvs'][0]
        print(f"  DGDV shape: {dgdv.shape} (nodes × orbits)")
        print(f"  Total orbits: {dgdv.shape[1]}")
        print(f"  Total orbit counts: {dgdv.sum()}")
        print(f"  Non-zero entries: {np.count_nonzero(dgdv)}")
    print(f"  Output: {output_dir}")
    print("="*60)


def normal_mode(project_root: Path):
    """Run pipeline in normal mode with full dataset"""
    pipeline_start = time.time()
    
    # Configuration
    data_dir = "data/malnet_tiny"
    output_dir = "data/gdvs"
    min_graphlet_size = 3  # UCL counter supports 2-4 node graphlets
    max_graphlet_size = 4  # Compute 3-node and 4-node DGDVs
    
    # Load dataset
    print("\n[1/3] Loading MalNet-Tiny dataset...")
    load_start = time.time()
    loader = MalNetTinyLoader(data_dir)
    graphs = loader.load_graphs(directed=True)  # Load as directed graphs
    load_time = time.time() - load_start
    
    if not graphs:
        print("[ERROR] No graphs loaded. Please check dataset directory.")
        return
    
    print(f"  Loaded {len(graphs)} graphs")
    if loader.labels:
        print(f"  Labels: {len(set(loader.labels))} unique families")
    print(f"  Loading time: {load_time:.2f} seconds")
    
    # Process graphs
    print(f"\n[2/3] Computing DGDVs (size: {min_graphlet_size}-{max_graphlet_size} node)...")
    dgdv_start = time.time()
    use_gpu = True  # Enable GPU acceleration if available
    processor = DGDVProcessor(data_dir, output_dir, use_gpu=use_gpu)
    results = processor.process_graphs(
        graphs,
        labels=loader.labels,
        min_graphlet_size=min_graphlet_size,
        max_graphlet_size=max_graphlet_size
    )
    dgdv_time = time.time() - dgdv_start
    
    # Summary
    total_time = time.time() - pipeline_start
    print(f"\n[3/3] Pipeline complete!")
    print(f"  Successful: {results['metadata']['successful']}")
    print(f"  Failed: {results['metadata']['failed']}")
    print(f"  Output: {output_dir}")
    print(f"\nTiming Summary:")
    print(f"  Dataset Loading: {load_time:.2f} seconds")
    print(f"  DGDV Computation: {dgdv_time:.2f} seconds")
    print(f"  Total Pipeline: {total_time:.2f} seconds")
    print("="*60)


if __name__ == "__main__":
    main()
