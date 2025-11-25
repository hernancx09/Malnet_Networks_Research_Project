#!/usr/bin/env python3
"""
Main pipeline for MalNet-Tiny graphlet analysis
Runs the complete workflow: Load -> Compute GDVs -> Save Results
"""

import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from load_malnet import MalNetTinyLoader
from compute_gdvs_malnet import GDVProcessor

def main():
    """Run the complete pipeline"""
    print("="*60)
    print("MalNet-Tiny Graphlet Analysis Pipeline")
    print("="*60)
    print("\nThis pipeline will:")
    print("1. Load MalNet-Tiny dataset")
    print("2. Compute Graphlet Degree Vectors (GDVs) for all graphs")
    print("3. Compute Graphlet Correlation Matrices (GCMs)")
    print("4. Save results for classification")
    print("\n" + "="*60)
    
    # Change to project directory
    os.chdir(project_root)
    
    # Configuration
    data_dir = "data/malnet_tiny"
    output_dir = "data/gdvs"
    max_graphlet_size = 4
    limit = None  # Set to a number to limit graphs for testing
    
    # Step 1: Load dataset
    print("\n[STEP 1] Loading MalNet-Tiny dataset...")
    loader = MalNetTinyLoader(data_dir)
    graphs = loader.load_graphs()
    
    if not graphs:
        print("\n[ERROR] No graphs loaded!")
        print("Please download MalNet-Tiny dataset first.")
        print("Run: python download_malnet.py")
        return
    
    print(f"[SUCCESS] Loaded {len(graphs)} graphs")
    
    # Get statistics
    stats = loader.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Total graphs: {stats['num_graphs']}")
    print(f"  Average nodes: {stats['avg_nodes']:.1f}")
    print(f"  Average edges: {stats['avg_edges']:.1f}")
    print(f"  Average density: {stats['avg_density']:.4f}")
    
    # Limit for testing if specified
    if limit:
        graphs = graphs[:limit]
        print(f"\n[INFO] Limited to {len(graphs)} graphs for processing")
    
    # Step 2: Compute GDVs
    print(f"\n[STEP 2] Computing GDVs (max_graphlet_size={max_graphlet_size})...")
    processor = GDVProcessor(data_dir, output_dir)
    results = processor.process_graphs(
        graphs, 
        max_graphlet_size=max_graphlet_size,
        save_individual=False
    )
    
    # Step 3: Summary
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - all_gdvs.pkl: All GDV matrices")
    print(f"  - all_gcms.pkl: All GCM vectors")
    print(f"  - metadata.json: Processing metadata")
    print("\nNext steps:")
    print("  1. Load GDVs/GCMs for classification")
    print("  2. Train ML models using GCM features")
    print("  3. Evaluate classification performance")
    print("="*60)


if __name__ == "__main__":
    main()



