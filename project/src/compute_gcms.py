#!/usr/bin/env python3
"""
GCM Computation Module
Computes Graphlet Correlation Matrices (GCMs) from DGDVs
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

# Import GPU acceleration if available
try:
    from helpers.gpu_acceleration import corrcoef as gpu_corrcoef
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    gpu_corrcoef = None


def compute_gcm_from_dgdv(dgdv: np.ndarray, use_gpu: bool = False) -> np.ndarray:
    """
    Compute GCM vector from a single DGDV matrix
    
    Args:
        dgdv: DGDV matrix of shape (n_nodes, n_orbits)
        use_gpu: Whether to use GPU acceleration if available
    
    Returns:
        GCM vector (flattened upper triangle of correlation matrix)
    """
    if dgdv.size == 0 or dgdv.shape[0] == 0:
        return np.array([])
    
    # Remove orbits with zero variance (they cause correlation issues)
    non_zero_orbits = np.var(dgdv, axis=0) > 0
    
    if non_zero_orbits.sum() == 0:
        return np.array([])
    
    dgdv_filtered = dgdv[:, non_zero_orbits]
    
    # Compute correlation matrix
    # Rows are nodes, columns are orbits
    # We want correlation between orbits (across nodes)
    try:
        if GPU_AVAILABLE and use_gpu and gpu_corrcoef is not None:
            corr_matrix = gpu_corrcoef(dgdv_filtered, use_gpu=True)
        else:
            # Compute correlation: transpose so orbits are rows
            # Each row is an orbit, each column is a node
            corr_matrix = np.corrcoef(dgdv_filtered.T)
        
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
    
    if batch_size is None:
        # Process all at once
        for dgdv in tqdm(dgdvs, desc="Computing GCMs"):
            gcm = compute_gcm_from_dgdv(dgdv, use_gpu=use_gpu)
            gcms.append(gcm)
    else:
        # Process in batches
        for batch_start in range(0, len(dgdvs), batch_size):
            batch_end = min(batch_start + batch_size, len(dgdvs))
            batch_dgdvs = dgdvs[batch_start:batch_end]
            
            for dgdv in batch_dgdvs:
                gcm = compute_gcm_from_dgdv(dgdv, use_gpu=use_gpu)
                gcms.append(gcm)
            
            if batch_end % (batch_size * 5) == 0:
                print(f"  Processed {batch_end}/{len(dgdvs)} GCMs...")
    
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
    with open(input_file, 'rb') as f:
        dgdvs = pickle.load(f)
    
    print(f"Loaded {len(dgdvs)} DGDVs")
    if len(dgdvs) > 0:
        print(f"  DGDV shape: {dgdvs[0].shape}")
    
    # Compute GCMs for remaining graphs
    if start_idx < len(dgdvs):
        remaining_dgdvs = dgdvs[start_idx:]
        print(f"\nComputing GCMs for {len(remaining_dgdvs)} graphs...")
        
        if GPU_AVAILABLE and use_gpu:
            print("  Using GPU acceleration")
        else:
            print("  Using CPU")
        
        new_gcms = compute_gcms_from_dgdvs(
            remaining_dgdvs,
            use_gpu=use_gpu,
            batch_size=batch_size
        )
        
        # Combine with existing
        all_gcms = existing_gcms + new_gcms
    else:
        all_gcms = existing_gcms
        print("\nAll GCMs already computed")
    
    # Save GCMs
    print(f"\nSaving GCMs to: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(all_gcms, f)
    
    file_size = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"  Saved {len(all_gcms)} GCMs ({file_size:.2f} MB)")
    
    # Statistics
    valid_gcms = [gcm for gcm in all_gcms if gcm.size > 0]
    if valid_gcms:
        gcm_sizes = [gcm.size for gcm in valid_gcms]
        print(f"\nGCM Statistics:")
        print(f"  Valid GCMs: {len(valid_gcms)}/{len(all_gcms)}")
        print(f"  Average GCM size: {np.mean(gcm_sizes):.1f}")
        print(f"  Min GCM size: {min(gcm_sizes)}")
        print(f"  Max GCM size: {max(gcm_sizes)}")
    
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

