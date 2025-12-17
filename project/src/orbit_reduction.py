#!/usr/bin/env python3
"""
Orbit Reduction Module
Removes orbits with near-zero variance or strong collinearity from DGDVs
"""

import numpy as np
import pickle
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr


def reduce_orbits(dgdvs: List[np.ndarray],
                  variance_threshold: float = 1e-6,
                  correlation_threshold: float = 0.95,
                  batch_size: int = 100) -> Tuple[List[np.ndarray], np.ndarray, Dict]:
    """
    Reduce orbits by removing near-zero variance and strongly collinear orbits
    
    Args:
        dgdvs: List of DGDV matrices, each of shape (n_nodes, n_orbits)
        variance_threshold: Minimum variance threshold (orbits below this are removed)
        correlation_threshold: Maximum correlation threshold (orbits above this are removed)
        batch_size: Batch size for processing large datasets
    
    Returns:
        Tuple of:
        - reduced_dgdvs: List of reduced DGDV matrices
        - kept_orbit_indices: Array of indices of kept orbits
        - reduction_stats: Dictionary with reduction statistics
    """
    if not dgdvs:
        return [], np.array([]), {}
    
    # Get dimensions
    n_graphs = len(dgdvs)
    n_orbits = dgdvs[0].shape[1]
    
    print(f"Reducing orbits from {n_orbits} to...")
    print(f"  Variance threshold: {variance_threshold}")
    print(f"  Correlation threshold: {correlation_threshold}")
    
    # Step 1: Remove orbits with near-zero variance
    # Compute variance across all graphs for each orbit
    print("\n[1/2] Computing orbit variances...")
    variance_start = time.time()
    orbit_variances = np.zeros(n_orbits)
    
    for batch_start in range(0, n_graphs, batch_size):
        batch_end = min(batch_start + batch_size, n_graphs)
        batch_dgdvs = dgdvs[batch_start:batch_end]
        
        # Stack all DGDVs in batch and compute variance per orbit
        stacked = np.vstack([dgdv for dgdv in batch_dgdvs if dgdv.size > 0])
        if stacked.size > 0:
            batch_variances = np.var(stacked, axis=0)
            orbit_variances = np.maximum(orbit_variances, batch_variances)
    
    variance_time = time.time() - variance_start
    print(f"  Variance computation time: {variance_time:.2f} seconds")
    
    # Find orbits with sufficient variance
    high_variance_mask = orbit_variances >= variance_threshold
    high_variance_indices = np.where(high_variance_mask)[0]
    
    print(f"  Orbits with variance >= {variance_threshold}: {len(high_variance_indices)}/{n_orbits}")
    print(f"  Removed {n_orbits - len(high_variance_indices)} low-variance orbits")
    
    if len(high_variance_indices) == 0:
        print("  WARNING: All orbits have low variance!")
        return dgdvs, np.arange(n_orbits), {
            'original_orbits': n_orbits,
            'after_variance_filter': n_orbits,
            'after_correlation_filter': n_orbits,
            'final_orbits': n_orbits,
            'variance_removed': 0,
            'correlation_removed': 0
        }
    
    # Step 2: Remove strongly collinear orbits
    print("\n[2/2] Computing orbit correlations...")
    correlation_start = time.time()
    
    # Compute mean orbit counts across all graphs for correlation analysis
    mean_orbit_counts = np.zeros((n_graphs, len(high_variance_indices)))
    
    for i, dgdv in enumerate(tqdm(dgdvs, desc="Computing mean orbit counts")):
        if dgdv.size > 0 and dgdv.shape[1] == n_orbits:
            # Take mean across nodes for each orbit
            mean_orbit_counts[i] = np.mean(dgdv[:, high_variance_indices], axis=0)
    
    # Compute correlation matrix of orbits
    # Remove any graphs with NaN or inf
    valid_mask = np.isfinite(mean_orbit_counts).all(axis=1)
    if valid_mask.sum() < 2:
        print("  WARNING: Not enough valid graphs for correlation analysis")
        kept_indices = high_variance_indices
    else:
        valid_means = mean_orbit_counts[valid_mask]
        corr_comp_start = time.time()
        correlation_matrix = np.corrcoef(valid_means.T)
        corr_comp_time = time.time() - corr_comp_start
        print(f"  Correlation matrix computation: {corr_comp_time:.2f} seconds")
        
        # Find strongly correlated orbit pairs
        # Use upper triangle to avoid duplicates
        n_kept = len(high_variance_indices)
        triu_indices = np.triu_indices(n_kept, k=1)
        correlations = correlation_matrix[triu_indices]
        
        # Find pairs with correlation > threshold
        high_corr_pairs = np.where(np.abs(correlations) > correlation_threshold)[0]
        
        if len(high_corr_pairs) > 0:
            # For each high-correlation pair, keep the orbit with higher variance
            to_remove = set()
            for pair_idx in high_corr_pairs:
                i, j = triu_indices[0][pair_idx], triu_indices[1][pair_idx]
                # Compare variances of these orbits
                var_i = orbit_variances[high_variance_indices[i]]
                var_j = orbit_variances[high_variance_indices[j]]
                # Remove the one with lower variance
                if var_i < var_j:
                    to_remove.add(i)
                else:
                    to_remove.add(j)
            
            # Create final mask
            final_mask = np.ones(len(high_variance_indices), dtype=bool)
            final_mask[list(to_remove)] = False
            kept_indices = high_variance_indices[final_mask]
            
            print(f"  Found {len(high_corr_pairs)} high-correlation pairs")
            print(f"  Removed {len(to_remove)} collinear orbits")
        else:
            kept_indices = high_variance_indices
            print(f"  No strongly correlated orbits found (threshold: {correlation_threshold})")
    
    correlation_time = time.time() - correlation_start
    print(f"  Correlation computation time: {correlation_time:.2f} seconds")
    
    print(f"\n  Final orbit count: {len(kept_indices)}/{n_orbits} ({len(kept_indices)/n_orbits*100:.1f}%)")
    
    # Step 3: Apply reduction to all DGDVs
    print("\n[3/3] Applying reduction to DGDVs...")
    apply_start = time.time()
    reduced_dgdvs = []
    
    for dgdv in tqdm(dgdvs, desc="Reducing DGDVs"):
        if dgdv.size > 0 and dgdv.shape[1] == n_orbits:
            reduced_dgdv = dgdv[:, kept_indices]
            reduced_dgdvs.append(reduced_dgdv)
        else:
            # Keep empty/invalid DGDVs as-is
            reduced_dgdvs.append(dgdv)
    
    apply_time = time.time() - apply_start
    print(f"  Reduction application time: {apply_time:.2f} seconds")
    
    # Compute statistics
    total_reduction_time = variance_time + correlation_time + apply_time
    reduction_stats = {
        'original_orbits': n_orbits,
        'after_variance_filter': len(high_variance_indices),
        'after_correlation_filter': len(kept_indices),
        'final_orbits': len(kept_indices),
        'variance_removed': n_orbits - len(high_variance_indices),
        'correlation_removed': len(high_variance_indices) - len(kept_indices),
        'reduction_ratio': len(kept_indices) / n_orbits,
        'variance_threshold': variance_threshold,
        'correlation_threshold': correlation_threshold,
        'kept_orbit_indices': kept_indices.tolist(),
        'timing': {
            'variance_computation': variance_time,
            'correlation_computation': correlation_time,
            'reduction_application': apply_time,
            'total': total_reduction_time
        }
    }
    
    print(f"\n  Total reduction time: {total_reduction_time:.2f} seconds")
    
    return reduced_dgdvs, kept_indices, reduction_stats


def reduce_orbits_from_file(input_file: str,
                           output_file: str,
                           metadata_file: Optional[str] = None,
                           variance_threshold: float = 1e-6,
                           correlation_threshold: float = 0.95,
                           batch_size: int = 100) -> Dict:
    """
    Load DGDVs from file, apply orbit reduction, and save results
    
    Args:
        input_file: Path to input DGDV pickle file
        output_file: Path to output reduced DGDV pickle file
        metadata_file: Path to save reduction metadata JSON (optional)
        variance_threshold: Minimum variance threshold
        correlation_threshold: Maximum correlation threshold
        batch_size: Batch size for processing
    
    Returns:
        Dictionary with reduction statistics
    """
    print("="*60)
    print("Orbit Reduction")
    print("="*60)
    
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
    
    # Apply reduction
    reduction_start = time.time()
    reduced_dgdvs, kept_indices, stats = reduce_orbits(
        dgdvs,
        variance_threshold=variance_threshold,
        correlation_threshold=correlation_threshold,
        batch_size=batch_size
    )
    reduction_time = time.time() - reduction_start
    
    # Save reduced DGDVs
    print(f"\nSaving reduced DGDVs to: {output_file}")
    save_start = time.time()
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(reduced_dgdvs, f)
    save_time = time.time() - save_start
    
    file_size = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"  Saved {len(reduced_dgdvs)} reduced DGDVs ({file_size:.2f} MB)")
    print(f"  Saving time: {save_time:.2f} seconds")
    
    # Save metadata
    if metadata_file:
        print(f"\nSaving metadata to: {metadata_file}")
        metadata_path = Path(metadata_file)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    # Timing summary
    total_time = load_time + reduction_time + save_time
    print(f"\nTiming Summary:")
    print(f"  Loading DGDVs: {load_time:.2f} seconds")
    print(f"  Reduction computation: {reduction_time:.2f} seconds")
    print(f"  Saving results: {save_time:.2f} seconds")
    print(f"  Total: {total_time:.2f} seconds")
    
    print("\n" + "="*60)
    print("Orbit Reduction Complete!")
    print("="*60)
    
    return stats


def reduce_orbits_yaveroglu(dgdvs: List[np.ndarray],
                            r2_threshold: float = 0.95,
                            batch_size: int = 100) -> Tuple[List[np.ndarray], np.ndarray, Dict]:
    """
    Reduce orbits based on Yaveroglu et al. methodology
    Identifies non-redundant orbits by finding linear dependencies
    
    The paper identifies redundancies through linear equations (e.g., C₀² = C₂ + C₃).
    This function adapts the methodology to directed graphlets by:
    1. Analyzing orbit count relationships across all graphs
    2. Using linear regression to identify which orbits can be predicted from others
    3. Keeping orbits that cannot be linearly derived from others
    
    Args:
        dgdvs: List of DGDV matrices, each of shape (n_nodes, n_orbits)
        r2_threshold: R² threshold for considering an orbit redundant (if R² > threshold, orbit is redundant)
        batch_size: Batch size for processing large datasets
    
    Returns:
        Tuple of:
        - reduced_dgdvs: List of reduced DGDV matrices
        - kept_orbit_indices: Array of indices of kept (non-redundant) orbits
        - reduction_stats: Dictionary with reduction statistics
    """
    if not dgdvs:
        return [], np.array([]), {}
    
    # Get dimensions
    n_graphs = len(dgdvs)
    n_orbits = dgdvs[0].shape[1]
    
    print(f"Yaveroglu-based orbit reduction: {n_orbits} orbits -> ...")
    print(f"  R² threshold: {r2_threshold}")
    
    # Step 1: Aggregate orbit counts across all graphs
    # For each graph, compute mean orbit counts (across nodes)
    print("\n[1/3] Aggregating orbit counts across graphs...")
    mean_orbit_counts = np.zeros((n_graphs, n_orbits))
    
    for i, dgdv in enumerate(tqdm(dgdvs, desc="Aggregating")):
        if dgdv.size > 0 and dgdv.shape[1] == n_orbits:
            # Take mean across nodes for each orbit
            mean_orbit_counts[i] = np.mean(dgdv, axis=0)
    
    # Remove orbits with zero variance
    orbit_variances = np.var(mean_orbit_counts, axis=0)
    non_zero_variance = orbit_variances > 1e-10
    valid_orbit_indices = np.where(non_zero_variance)[0]
    
    if len(valid_orbit_indices) == 0:
        print("  WARNING: All orbits have zero variance!")
        return dgdvs, np.arange(n_orbits), {
            'original_orbits': n_orbits,
            'final_orbits': n_orbits,
            'redundant_orbits': 0
        }
    
    print(f"  Valid orbits (non-zero variance): {len(valid_orbit_indices)}/{n_orbits}")
    
    # Step 2: Identify redundant orbits using linear regression
    print("\n[2/3] Identifying redundant orbits via linear dependencies...")
    
    valid_means = mean_orbit_counts[:, valid_orbit_indices]
    n_valid = len(valid_orbit_indices)
    
    # For each orbit, try to predict it from other orbits
    redundant_mask = np.zeros(n_valid, dtype=bool)
    
    for i in tqdm(range(n_valid), desc="Analyzing dependencies"):
        # Target: orbit i
        y = valid_means[:, i]
        
        # Features: all other orbits
        X = np.delete(valid_means, i, axis=1)
        
        # Skip if target has no variance
        if np.var(y) < 1e-10:
            redundant_mask[i] = True
            continue
        
        # Fit linear regression
        try:
            reg = LinearRegression()
            reg.fit(X, y)
            y_pred = reg.predict(X)
            
            # Compute R²
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # If R² is high, orbit is redundant
            if r2 > r2_threshold:
                redundant_mask[i] = True
        except:
            # If regression fails, keep the orbit
            pass
    
    # Get non-redundant orbits
    non_redundant_mask = ~redundant_mask
    non_redundant_local_indices = np.where(non_redundant_mask)[0]
    kept_orbit_indices = valid_orbit_indices[non_redundant_local_indices]
    
    n_redundant = redundant_mask.sum()
    print(f"  Redundant orbits: {n_redundant}")
    print(f"  Non-redundant orbits: {len(kept_orbit_indices)}/{n_orbits}")
    
    # Step 3: Apply reduction to all DGDVs
    print("\n[3/3] Applying reduction to DGDVs...")
    reduced_dgdvs = []
    
    for dgdv in tqdm(dgdvs, desc="Reducing DGDVs"):
        if dgdv.size > 0 and dgdv.shape[1] == n_orbits:
            reduced_dgdv = dgdv[:, kept_orbit_indices]
            reduced_dgdvs.append(reduced_dgdv)
        else:
            # Keep empty/invalid DGDVs as-is
            reduced_dgdvs.append(dgdv)
    
    # Compute statistics
    reduction_stats = {
        'original_orbits': n_orbits,
        'valid_orbits': len(valid_orbit_indices),
        'final_orbits': len(kept_orbit_indices),
        'redundant_orbits': n_redundant,
        'reduction_ratio': len(kept_orbit_indices) / n_orbits,
        'r2_threshold': r2_threshold,
        'kept_orbit_indices': kept_orbit_indices.tolist()
    }
    
    print(f"\n  Final orbit count: {len(kept_orbit_indices)}/{n_orbits} ({len(kept_orbit_indices)/n_orbits*100:.1f}%)")
    
    return reduced_dgdvs, kept_orbit_indices, reduction_stats


def reduce_orbits_yaveroglu_from_file(input_file: str,
                                      output_file: str,
                                      metadata_file: Optional[str] = None,
                                      r2_threshold: float = 0.95,
                                      batch_size: int = 100) -> Dict:
    """
    Load DGDVs from file, apply Yaveroglu-based orbit reduction, and save results
    
    Args:
        input_file: Path to input DGDV pickle file
        output_file: Path to output reduced DGDV pickle file
        metadata_file: Path to save reduction metadata JSON (optional)
        r2_threshold: R² threshold for considering an orbit redundant
        batch_size: Batch size for processing
    
    Returns:
        Dictionary with reduction statistics
    """
    print("="*60)
    print("Yaveroglu-based Orbit Reduction")
    print("="*60)
    
    # Load DGDVs
    print(f"\nLoading DGDVs from: {input_file}")
    with open(input_file, 'rb') as f:
        dgdvs = pickle.load(f)
    
    print(f"Loaded {len(dgdvs)} DGDVs")
    if len(dgdvs) > 0:
        print(f"  DGDV shape: {dgdvs[0].shape}")
    
    # Apply reduction
    reduced_dgdvs, kept_indices, stats = reduce_orbits_yaveroglu(
        dgdvs,
        r2_threshold=r2_threshold,
        batch_size=batch_size
    )
    
    # Save reduced DGDVs
    print(f"\nSaving reduced DGDVs to: {output_file}")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(reduced_dgdvs, f)
    
    file_size = Path(output_file).stat().st_size / (1024 * 1024)
    print(f"  Saved {len(reduced_dgdvs)} reduced DGDVs ({file_size:.2f} MB)")
    
    # Save metadata
    if metadata_file:
        print(f"\nSaving metadata to: {metadata_file}")
        metadata_path = Path(metadata_file)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_file, 'w') as f:
            json.dump(stats, f, indent=2)
    
    print("\n" + "="*60)
    print("Yaveroglu-based Orbit Reduction Complete!")
    print("="*60)
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Reduce orbits in DGDVs')
    parser.add_argument('--input', type=str, default='data/gdvs/gdvs/all_dgdvs_3_4node.pkl',
                       help='Input DGDV file')
    parser.add_argument('--output', type=str, default='data/gdvs/gdvs/all_dgdvs_3_4node_reduced.pkl',
                       help='Output reduced DGDV file')
    parser.add_argument('--metadata', type=str, default='data/gdvs/metadata/orbit_reduction_3_4node.json',
                       help='Output metadata file')
    parser.add_argument('--variance-threshold', type=float, default=1e-6,
                       help='Variance threshold for orbit removal')
    parser.add_argument('--correlation-threshold', type=float, default=0.95,
                       help='Correlation threshold for orbit removal')
    parser.add_argument('--yaveroglu', action='store_true',
                       help='Use Yaveroglu-based reduction method')
    parser.add_argument('--r2-threshold', type=float, default=0.95,
                       help='R² threshold for Yaveroglu method')
    
    args = parser.parse_args()
    
    if args.yaveroglu:
        reduce_orbits_yaveroglu_from_file(
            args.input,
            args.output,
            args.metadata,
            r2_threshold=args.r2_threshold
        )
    else:
        reduce_orbits_from_file(
            args.input,
            args.output,
            args.metadata,
            variance_threshold=args.variance_threshold,
            correlation_threshold=args.correlation_threshold
        )

