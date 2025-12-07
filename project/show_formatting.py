#!/usr/bin/env python3
"""
Display formatting and structure of DGDVs and GCMs
"""

import pickle
import numpy as np
from pathlib import Path

print("="*80)
print("DGDV and GCM Formatting")
print("="*80)

# Use known paths - try different file names
dgdv_dir = Path("data/DGDVs")
gcm_dir = Path("data/GCMs")

# Find DGDV files - check for combined or use individual
dgdv_file = None
dgdv_individual = dgdv_dir / "individual" if dgdv_dir.exists() else None

if dgdv_dir.exists():
    # Look for combined files
    dgdv_files = list(dgdv_dir.glob("*.pkl"))
    if dgdv_files:
        reduced = [f for f in dgdv_files if "reduced" in f.name]
        dgdv_file = reduced[0] if reduced else dgdv_files[0]
    # If no combined file, we'll use individual files
    elif dgdv_individual and dgdv_individual.exists():
        individual_files = sorted(dgdv_individual.glob("graph_*.pkl"))
        if individual_files:
            # Use first individual file as sample
            dgdv_file = individual_files[0]
            print(f"Note: Using individual file as sample: {dgdv_file.name}")

# Find GCM files - check for combined or use individual  
gcm_file = None
gcm_individual = gcm_dir / "individual" if gcm_dir.exists() else None

if gcm_dir.exists():
    # Look for combined files
    gcm_files = list(gcm_dir.glob("*.pkl"))
    if gcm_files:
        reduced = [f for f in gcm_files if "reduced" in f.name]
        gcm_file = reduced[0] if reduced else gcm_files[0]
    # If no combined file, we'll use individual files
    elif gcm_individual and gcm_individual.exists():
        individual_files = sorted(gcm_individual.glob("graph_*.pkl"))
        if individual_files:
            # Use first individual file as sample
            gcm_file = individual_files[0]
            print(f"Note: Using individual file as sample: {gcm_file.name}")

# ============================================================================
# DGDV FORMATTING
# ============================================================================
print("\n" + "="*80)
print("DGDV (Directed Graphlet Degree Vector) FORMATTING")
print("="*80)

if dgdv_file and dgdv_file.exists():
    # Check if it's a combined file (list) or individual file (array)
    with open(dgdv_file, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, list):
        # Combined file
        print(f"\n📁 Combined File: {dgdv_file}")
        dgdvs = data
        print(f"\n📊 Combined File Structure:")
        print(f"  Type: {type(dgdvs)}")
        print(f"  Number of graphs: {len(dgdvs)}")
        sample_dgdv = dgdvs[0] if len(dgdvs) > 0 else None
    else:
        # Individual file
        print(f"\n📁 Individual File (Sample): {dgdv_file.name}")
        sample_dgdv = data
        dgdvs = [data]  # For compatibility with rest of code
    
    if sample_dgdv is not None:
        print(f"\n📐 Sample DGDV (first graph):")
        print(f"  Type: {type(sample_dgdv)}")
        print(f"  Shape: {sample_dgdv.shape}")
        print(f"  Dtype: {sample_dgdv.dtype}")
        print(f"  Size: {sample_dgdv.size} elements")
        print(f"  Memory: {sample_dgdv.nbytes / 1024:.2f} KB")
        print(f"  Interpretation: {sample_dgdv.shape[0]} nodes × {sample_dgdv.shape[1]} orbits")
        
        print(f"\n  Content Preview:")
        print(f"    First row (first node, all orbits):")
        print(f"      {sample_dgdv[0, :15]} ... (showing first 15 of {sample_dgdv.shape[1]} orbits)")
        print(f"    First column (all nodes, first orbit):")
        print(f"      {sample_dgdv[:15, 0]} ... (showing first 15 of {sample_dgdv.shape[0]} nodes)")
        
        print(f"\n  Statistics:")
        print(f"    Min value: {sample_dgdv.min()}")
        print(f"    Max value: {sample_dgdv.max()}")
        print(f"    Mean value: {sample_dgdv.mean():.2f}")
        print(f"    Non-zero entries: {(sample_dgdv > 0).sum()} / {sample_dgdv.size} ({100*(sample_dgdv > 0).sum()/sample_dgdv.size:.1f}%)")
        
        # Show orbit distribution
        orbit_sums = sample_dgdv.sum(axis=0)
        print(f"\n  Orbit Activity (sum across all nodes):")
        print(f"    Active orbits (sum > 0): {(orbit_sums > 0).sum()} / {len(orbit_sums)}")
        top_orbits = np.argsort(orbit_sums)[-5:][::-1]
        print(f"    Top 5 most active orbits: {top_orbits}")
        print(f"    Their counts: {orbit_sums[top_orbits]}")
        
        # Show a few more examples if we have a combined file
        if isinstance(data, list) and len(dgdvs) > 1:
            print(f"\n  Additional Examples:")
            for i in [1, 2, 3]:
                if i < len(dgdvs):
                    dgdv = dgdvs[i]
                    print(f"    Graph {i}: shape={dgdv.shape}, nodes={dgdv.shape[0]}, orbits={dgdv.shape[1]}")
    
    # Show individual files info
    if dgdv_individual and dgdv_individual.exists():
        individual_files = sorted(dgdv_individual.glob("graph_*.pkl"))
        if individual_files:
            print(f"\n📁 Individual Files:")
            print(f"  Total individual files: {len(individual_files)}")
            print(f"  Location: {dgdv_individual}")
            if not isinstance(data, list):
                # We already showed the sample
                print(f"  Sample file shown above: {dgdv_file.name}")
            else:
                # Show a sample individual file
                print(f"  Sample: {individual_files[0].name}")
                with open(individual_files[0], 'rb') as f:
                    ind_dgdv = pickle.load(f)
                print(f"    Shape: {ind_dgdv.shape}, Dtype: {ind_dgdv.dtype}")
                print(f"    Matches combined[0]: {np.array_equal(dgdvs[0], ind_dgdv)}")
else:
    print(f"\n⚠️  DGDV file not found at: {dgdv_file}")

# ============================================================================
# GCM FORMATTING
# ============================================================================
print("\n" + "="*80)
print("GCM (Graphlet Correlation Matrix) FORMATTING")
print("="*80)

if gcm_file and gcm_file.exists():
    # Check if it's a combined file (list) or individual file (array)
    with open(gcm_file, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, list):
        # Combined file
        print(f"\n📁 Combined File: {gcm_file}")
        gcms = data
        print(f"\n📊 Combined File Structure:")
        print(f"  Type: {type(gcms)}")
        print(f"  Number of graphs: {len(gcms)}")
        # Find a valid (non-empty) GCM
        valid_gcms = [g for g in gcms if hasattr(g, 'size') and g.size > 0]
        if valid_gcms:
            sample_gcm = valid_gcms[0]
            sample_idx = gcms.index(sample_gcm)
        else:
            sample_gcm = None
    else:
        # Individual file
        print(f"\n📁 Individual File (Sample): {gcm_file.name}")
        sample_gcm = data if hasattr(data, 'size') and data.size > 0 else None
        gcms = [data] if sample_gcm is not None else []
        valid_gcms = [sample_gcm] if sample_gcm is not None else []
        sample_idx = 0
    
    if sample_gcm is not None:
        print(f"\n📐 Sample GCM (graph {sample_idx}):")
        print(f"  Type: {type(sample_gcm)}")
        print(f"  Shape: {sample_gcm.shape}")
        print(f"  Dtype: {sample_gcm.dtype}")
        print(f"  Size: {sample_gcm.size} elements")
        print(f"  Memory: {sample_gcm.nbytes / 1024:.2f} KB")
        
        # Reconstruct original correlation matrix dimensions
        # GCM size = n*(n-1)/2, so n = (1 + sqrt(1 + 8*size)) / 2
        n_orbits = int((1 + np.sqrt(1 + 8 * sample_gcm.size)) / 2)
        print(f"  Original correlation matrix: {n_orbits} × {n_orbits} orbits")
        print(f"  GCM contains: upper triangle (excluding diagonal)")
        print(f"    Formula: n × (n-1) / 2 = {n_orbits} × {n_orbits-1} / 2 = {sample_gcm.size}")
        
        print(f"\n  Content Preview:")
        print(f"    First 20 correlation values:")
        print(f"      {sample_gcm[:20]}")
        print(f"    Last 10 correlation values:")
        print(f"      {sample_gcm[-10:]}")
        
        print(f"\n  Statistics:")
        print(f"    Min correlation: {sample_gcm.min():.6f}")
        print(f"    Max correlation: {sample_gcm.max():.6f}")
        print(f"    Mean correlation: {sample_gcm.mean():.6f}")
        print(f"    Std deviation: {sample_gcm.std():.6f}")
        print(f"    Median: {np.median(sample_gcm):.6f}")
        
        # Correlation distribution
        positive = (sample_gcm > 0).sum()
        negative = (sample_gcm < 0).sum()
        zero = (sample_gcm == 0).sum()
        strong_pos = (sample_gcm > 0.5).sum()
        strong_neg = (sample_gcm < -0.5).sum()
        moderate_pos = ((sample_gcm > 0.3) & (sample_gcm <= 0.5)).sum()
        moderate_neg = ((sample_gcm < -0.3) & (sample_gcm >= -0.5)).sum()
        
        print(f"\n  Correlation Distribution:")
        print(f"    Positive: {positive} ({100*positive/sample_gcm.size:.1f}%)")
        print(f"    Negative: {negative} ({100*negative/sample_gcm.size:.1f}%)")
        print(f"    Zero: {zero} ({100*zero/sample_gcm.size:.1f}%)")
        print(f"    Strong positive (>0.5): {strong_pos} ({100*strong_pos/sample_gcm.size:.1f}%)")
        print(f"    Moderate positive (0.3-0.5): {moderate_pos} ({100*moderate_pos/sample_gcm.size:.1f}%)")
        print(f"    Moderate negative (-0.5 to -0.3): {moderate_neg} ({100*moderate_neg/sample_gcm.size:.1f}%)")
        print(f"    Strong negative (<-0.5): {strong_neg} ({100*strong_neg/sample_gcm.size:.1f}%)")
        
        # Show how to reconstruct full matrix
        print(f"\n  Matrix Reconstruction Example:")
        print(f"    To reconstruct full {n_orbits}×{n_orbits} correlation matrix:")
        print(f"      corr_matrix = np.zeros(({n_orbits}, {n_orbits}))")
        print(f"      triu_indices = np.triu_indices({n_orbits}, k=1)")
        print(f"      corr_matrix[triu_indices] = gcm")
        print(f"      corr_matrix.T[triu_indices] = gcm  # Make symmetric")
        print(f"      np.fill_diagonal(corr_matrix, 1.0)  # Self-correlation")
        
        # Show a few more examples if we have a combined file
        if isinstance(data, list) and len(valid_gcms) > 1:
            print(f"\n  Additional Examples:")
            for i, gcm in enumerate(valid_gcms[1:4], 1):
                n = int((1 + np.sqrt(1 + 8 * gcm.size)) / 2)
                print(f"    Graph {sample_idx + i}: size={gcm.size}, {n}×{n} matrix, "
                      f"mean={gcm.mean():.3f}, range=[{gcm.min():.3f}, {gcm.max():.3f}]")
    
    # Show size distribution if we have multiple GCMs
    if isinstance(data, list):
            gcm_sizes = [g.size for g in gcms if hasattr(g, 'size') and g.size > 0]
            if gcm_sizes:
                print(f"\n  GCM Size Distribution (across all {len(gcm_sizes)} valid graphs):")
                print(f"    Min size: {min(gcm_sizes)}")
                print(f"    Max size: {max(gcm_sizes)}")
                print(f"    Mean size: {np.mean(gcm_sizes):.1f}")
                print(f"    Median size: {np.median(gcm_sizes):.1f}")
                print(f"    Std deviation: {np.std(gcm_sizes):.1f}")
    
    # Show individual files info
    if gcm_individual and gcm_individual.exists():
        individual_files = sorted(gcm_individual.glob("graph_*.pkl"))
        if individual_files:
            print(f"\n📁 Individual Files:")
            print(f"  Total individual files: {len(individual_files)}")
            print(f"  Location: {gcm_individual}")
            if not isinstance(data, list):
                # We already showed the sample
                print(f"  Sample file shown above: {gcm_file.name}")
            else:
                # Show a sample individual file
                print(f"  Sample: {individual_files[0].name}")
                with open(individual_files[0], 'rb') as f:
                    ind_gcm = pickle.load(f)
                print(f"    Shape: {ind_gcm.shape}, Dtype: {ind_gcm.dtype}")
                if valid_gcms:
                    print(f"    Matches combined[0]: {np.array_equal(gcms[0], ind_gcm)}")
else:
    print(f"\n⚠️  GCM file not found at: {gcm_file}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "="*80)
print("DGDV vs GCM COMPARISON")
print("="*80)

# Initialize variables for comparison
sample_dgdv = None
sample_gcm = None

if dgdv_file and dgdv_file.exists():
    with open(dgdv_file, 'rb') as f:
        dgdv_data = pickle.load(f)
    if isinstance(dgdv_data, list) and len(dgdv_data) > 0:
        sample_dgdv = dgdv_data[0]
    elif hasattr(dgdv_data, 'shape'):
        sample_dgdv = dgdv_data

if gcm_file and gcm_file.exists():
    with open(gcm_file, 'rb') as f:
        gcm_data = pickle.load(f)
    if isinstance(gcm_data, list):
        valid = [g for g in gcm_data if hasattr(g, 'size') and g.size > 0]
        if valid:
            sample_gcm = valid[0]
    elif hasattr(gcm_data, 'size') and gcm_data.size > 0:
        sample_gcm = gcm_data

if sample_dgdv is not None and sample_gcm is not None:
    print(f"\n📊 Key Differences:")
    print(f"\n  DGDV (Directed Graphlet Degree Vector):")
    print(f"    - Level: Node-level features")
    print(f"    - Structure: One matrix per graph")
    print(f"    - Shape: (n_nodes, n_orbits)")
    print(f"    - Contains: Orbit counts for each node")
    print(f"    - Example: {sample_dgdv.shape}")
    print(f"    - Meaning: Each row = one node, each column = one orbit")
    print(f"    - Values: Non-negative integers (counts)")
    print(f"    - Data type: {sample_dgdv.dtype}")
    
    print(f"\n  GCM (Graphlet Correlation Matrix):")
    print(f"    - Level: Graph-level features")
    print(f"    - Structure: One vector per graph")
    print(f"    - Shape: (n_correlations,) - flattened upper triangle")
    print(f"    - Contains: Correlations between orbit pairs")
    print(f"    - Example: {sample_gcm.shape}")
    print(f"    - Meaning: Each element = correlation between two orbits")
    print(f"    - Values: Floats in range [-1, 1] (correlation coefficients)")
    print(f"    - Data type: {sample_gcm.dtype}")
    
    print(f"\n  Relationship:")
    print(f"    - GCM is computed FROM DGDV")
    print(f"    - Process:")
    print(f"      1. Start with DGDV: (n_nodes, n_orbits) matrix")
    print(f"      2. Compute correlation between orbits (across nodes)")
    print(f"      3. Result: (n_orbits, n_orbits) correlation matrix")
    print(f"      4. Extract upper triangle (excluding diagonal)")
    print(f"      5. Flatten to vector: n_orbits × (n_orbits-1) / 2 elements")
    print(f"    - One DGDV matrix → One GCM vector")
    print(f"    - DGDV preserves node-level information")
    print(f"    - GCM captures orbit interaction patterns")

print("\n" + "="*80)

