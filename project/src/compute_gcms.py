#!/usr/bin/env python3
"""
GCM Computation Module
Computes Graphlet Correlation Matrices (GCMs) from DGDVs
Based on Yaveroglu et al. methodology using Spearman correlation
"""

import numpy as np
import pickle
import time
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
from scipy.stats import spearmanr

# Import GPU acceleration if available
try:
    from helpers.gpu_acceleration import corrcoef as gpu_corrcoef
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    gpu_corrcoef = None


def compute_gcm_from_dgdv(dgdv: np.ndarray, use_gpu: bool = False) -> np.ndarray:
    """
    Compute GCM vector from a single DGDV matrix using Spearman correlation
    Based on Yaveroglu et al. methodology
    
    Includes both 3-node and 4-node orbits:
    - 3-node orbits: 2-40 (39 orbits)
    - 4-node orbits: 41-128 (88 orbits)
    
    Args:
        dgdv: DGDV matrix of shape (n_nodes, n_orbits)
        use_gpu: Whether to use GPU acceleration if available (not used for Spearman)
    
    Returns:
        GCM vector (flattened upper triangle of Spearman correlation matrix)
    """
    if dgdv.size == 0 or dgdv.shape[0] == 0:
        return np.array([])
    
    # Remove orbits with zero variance (they cause correlation issues)
    non_zero_orbits = np.var(dgdv, axis=0) > 0
    
    if non_zero_orbits.sum() == 0:
        return np.array([])
    
    dgdv_filtered = dgdv[:, non_zero_orbits]
    n_orbits = dgdv_filtered.shape[1]
    
    # Compute Spearman correlation matrix
    # Rows are nodes, columns are orbits
    # We want Spearman correlation between orbits (across nodes)
    try:
        # Use Spearman correlation (rank-based) instead of Pearson
        # For each pair of orbits, compute Spearman correlation
        corr_matrix = np.eye(n_orbits)  # Initialize with identity (diagonal = 1.0)
        
        for i in range(n_orbits):
            for j in range(i + 1, n_orbits):
                # Compute Spearman correlation between orbit i and orbit j
                corr, _ = spearmanr(dgdv_filtered[:, i], dgdv_filtered[:, j])
                # Handle NaN (can occur if all values are the same)
                if np.isnan(corr):
                    corr = 0.0
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr  # Make symmetric
        
        if corr_matrix.size == 0:
            return np.array([])
        
        # Extract upper triangular portion (excluding diagonal)
        n = corr_matrix.shape[0]
        gcm = corr_matrix[np.triu_indices(n, k=1)]
        
        return gcm
        
    except Exception as e:
        # Return empty array on error
        print(f"Warning: Error computing GCM: {e}")
        return np.array([])


def compute_gcms_from_dgdvs(dgdvs: List[np.ndarray],
                           use_gpu: bool = False,
                           batch_size: Optional[int] = None) -> List[np.ndarray]:
    """
    Compute GCMs from a list of DGDV matrices
    
    Args:
        dgdvs: List of DGDV matrices
        use_gpu: Whether to use GPU acceleration if available
        batch_size: Batch size for processing (None = process all at once)
    
    Returns:
        List of GCM vectors
    """
    gcms = []
    comp_times = []
    
    if batch_size is None:
        # Process all at once
        for i, dgdv in enumerate(tqdm(dgdvs, desc="Computing GCMs")):
            comp_start = time.time()
            gcm = compute_gcm_from_dgdv(dgdv, use_gpu=use_gpu)
            comp_time = time.time() - comp_start
            comp_times.append(comp_time)
            gcms.append(gcm)
            
            if (i + 1) % 100 == 0 and i > 0:
                avg_time = np.mean(comp_times[-100:])
                print(f"  Graph {i+1}/{len(dgdvs)}: avg {avg_time:.3f}s/graph")
    else:
        # Process in batches
        for batch_start in range(0, len(dgdvs), batch_size):
            batch_end = min(batch_start + batch_size, len(dgdvs))
            batch_dgdvs = dgdvs[batch_start:batch_end]
            
            for dgdv in batch_dgdvs:
                comp_start = time.time()
                gcm = compute_gcm_from_dgdv(dgdv, use_gpu=use_gpu)
                comp_time = time.time() - comp_start
                comp_times.append(comp_time)
                gcms.append(gcm)
            
            if batch_end % (batch_size * 5) == 0:
                avg_time = np.mean(comp_times[-batch_size*5:]) if comp_times else 0
                print(f"  Processed {batch_end}/{len(dgdvs)} GCMs... (avg: {avg_time:.3f}s/graph)")
    
    return gcms


def compute_gcms_from_file(input_file: str,
                          output_file: str,
                          use_gpu: bool = False,
                          batch_size: int = 100,
                          resume: bool = True) -> None:
    """
    Load DGDVs from file, compute GCMs, and save results
    
    Args:
        input_file: Path to input DGDV pickle file
        output_file: Path to output GCM pickle file
        use_gpu: Whether to use GPU acceleration if available
        batch_size: Batch size for processing
        resume: If True, check for existing GCMs and resume
    """
    print("="*60)
    print("GCM Computation")
    print("="*60)
    
    # Check for existing GCMs if resuming
    existing_gcms = []
    start_idx = 0
    
    if resume and Path(output_file).exists():
        try:
            print(f"\nFound existing GCM file: {output_file}")
            with open(output_file, 'rb') as f:
                existing_gcms = pickle.load(f)
            start_idx = len(existing_gcms)
            print(f"  Resuming from index {start_idx}")
        except Exception as e:
            print(f"  Warning: Could not load existing GCMs: {e}")
            existing_gcms = []
            start_idx = 0
    
    # Load DGDVs
    print(f"\nLoading DGDVs from: {input_file}")
    load_start = time.time()
    with open(input_file, 'rb') as f:
        dgdvs = pickle.load(f)
    load_time = time.time() - load_start
    print(f"Loaded {len(dgdvs)} DGDVs")
    if len(dgdvs) > 0:
        print(f"  DGDV shape: {dgdvs[0].shape}")
    print(f"  Loading time: {load_time:.2f} seconds")
    
    # Compute GCMs for remaining graphs
    if start_idx < len(dgdvs):
        remaining_dgdvs = dgdvs[start_idx:]
        print(f"\nComputing GCMs for {len(remaining_dgdvs)} graphs...")
        
        if GPU_AVAILABLE and use_gpu:
            print("  Using GPU acceleration")
        else:
            print("  Using CPU")
        
        comp_start = time.time()
        new_gcms = compute_gcms_from_dgdvs(
            remaining_dgdvs,
            use_gpu=use_gpu,
            batch_size=batch_size
        )
        comp_time = time.time() - comp_start
        
        # Combine with existing
        all_gcms = existing_gcms + new_gcms
    else:
        all_gcms = existing_gcms
        print("\nAll GCMs already computed")
    
    # Save GCMs
    print(f"\nSaving GCMs to: {output_file}")
    save_start = time.time()
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(all_gcms, f)
    save_time = time.time() - save_start
    
    file_size = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"  Saved {len(all_gcms)} GCMs ({file_size:.2f} MB)")
    print(f"  Saving time: {save_time:.2f} seconds")
    
    # Statistics
    valid_gcms = [gcm for gcm in all_gcms if gcm.size > 0]
    if valid_gcms:
        gcm_sizes = [gcm.size for gcm in valid_gcms]
        print(f"\nGCM Statistics:")
        print(f"  Valid GCMs: {len(valid_gcms)}/{len(all_gcms)}")
        print(f"  Average GCM size: {np.mean(gcm_sizes):.1f}")
        print(f"  Min GCM size: {min(gcm_sizes)}")
        print(f"  Max GCM size: {max(gcm_sizes)}")
    
    # Timing summary
    total_time = load_time + (comp_time if start_idx < len(dgdvs) else 0) + save_time
    print(f"\nTiming Summary:")
    print(f"  Loading DGDVs: {load_time:.2f} seconds")
    if start_idx < len(dgdvs):
        print(f"  Computing GCMs: {comp_time:.2f} seconds")
        if len(remaining_dgdvs) > 0:
            print(f"  Average: {comp_time/len(remaining_dgdvs):.2f} seconds/graph")
    print(f"  Saving GCMs: {save_time:.2f} seconds")
    print(f"  Total: {total_time:.2f} seconds")
    
    print("\n" + "="*60)
    print("GCM Computation Complete!")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute GCMs from DGDVs')
    parser.add_argument('--input', type=str, default='data/gdvs/gdvs/all_dgdvs_3_4node_reduced.pkl',
                       help='Input DGDV file')
    parser.add_argument('--output', type=str, default='data/gdvs/gcms/all_gcms_3_4node_reduced.pkl',
                       help='Output GCM file')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU acceleration if available')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for processing')
    parser.add_argument('--no-resume', action='store_true',
                       help='Do not resume from existing GCMs')
    
    args = parser.parse_args()
    
    compute_gcms_from_file(
        args.input,
        args.output,
        use_gpu=args.gpu,
        batch_size=args.batch_size,
        resume=not args.no_resume
    )

