#!/usr/bin/env python3
"""
Feature Extraction Module
Extracts coarse features from DGDVs and loads GCM features
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import sys

# Add parent directory to path to import load_malnet
sys.path.insert(0, str(Path(__file__).parent.parent))

from load_malnet import MalNetTinyLoader


def extract_coarse_features(dgdvs: List[np.ndarray]) -> np.ndarray:
    """
    Extract coarse features from DGDV matrices
    
    For each DGDV, compute aggregate statistics per orbit:
    - Mean across nodes (per orbit)
    - Std deviation across nodes (per orbit)
    - Sum across nodes (per orbit)
    - Min across nodes (per orbit)
    - Max across nodes (per orbit)
    
    Args:
        dgdvs: List of DGDV matrices, each of shape (n_nodes, n_orbits)
    
    Returns:
        Feature matrix of shape (n_graphs, n_orbits × 5)
    """
    if not dgdvs:
        return np.array([])
    
    features = []
    
    for dgdv in dgdvs:
        if dgdv.size == 0 or dgdv.shape[0] == 0:
            # Empty graph - use zeros with correct shape
            if features:
                n_orbits = features[0].shape[0] // 5
                graph_features = np.zeros(n_orbits * 5)
            else:
                # First empty graph - can't determine shape, skip
                continue
        else:
            # Compute statistics per orbit (across nodes)
            means = np.mean(dgdv, axis=0)      # Mean per orbit
            stds = np.std(dgdv, axis=0)        # Std per orbit
            sums = np.sum(dgdv, axis=0)        # Sum per orbit
            mins = np.min(dgdv, axis=0)        # Min per orbit
            maxs = np.max(dgdv, axis=0)        # Max per orbit
            
            # Concatenate statistics
            graph_features = np.concatenate([means, stds, sums, mins, maxs])
        
        features.append(graph_features)
    
    if not features:
        return np.array([])
    
    return np.array(features)


def load_gcm_features(gcm_file: str) -> np.ndarray:
    """
    Load GCM vectors from pickle file
    
    Args:
        gcm_file: Path to GCM pickle file (list of GCM vectors)
    
    Returns:
        Feature matrix of shape (n_graphs, n_gcm_features)
    """
    gcm_path = Path(gcm_file)
    
    if not gcm_path.exists():
        raise FileNotFoundError(f"GCM file not found: {gcm_file}")
    
    with open(gcm_path, 'rb') as f:
        gcms = pickle.load(f)
    
    if isinstance(gcms, list):
        # List of arrays - handle different shapes
        # Filter out empty arrays and find max size
        valid_gcms = [gcm for gcm in gcms if gcm.size > 0]
        
        if not valid_gcms:
            raise ValueError("No valid GCM vectors found in file")
        
        # Find the maximum size
        max_size = max(gcm.size for gcm in valid_gcms)
        
        # Pad or truncate all GCMs to the same size
        normalized_gcms = []
        for gcm in gcms:
            if gcm.size == 0:
                # Empty GCM - pad with zeros
                normalized_gcm = np.zeros(max_size)
            elif gcm.size < max_size:
                # Pad with zeros
                normalized_gcm = np.pad(gcm, (0, max_size - gcm.size), mode='constant')
            elif gcm.size > max_size:
                # Truncate (shouldn't happen, but handle it)
                normalized_gcm = gcm[:max_size]
            else:
                normalized_gcm = gcm
            
            normalized_gcms.append(normalized_gcm)
        
        return np.array(normalized_gcms)
    elif isinstance(gcms, np.ndarray):
        # Already a numpy array
        if gcms.ndim == 1:
            # Single GCM vector - reshape to (1, n_features)
            return gcms.reshape(1, -1)
        return gcms
    else:
        raise ValueError(f"Unexpected GCM file format. Expected list or array, got {type(gcms)}")


def load_labels_from_loader(loader: MalNetTinyLoader) -> Tuple[np.ndarray, Dict]:
    """
    Extract labels from loader and encode them as integers
    
    Args:
        loader: MalNetTinyLoader instance with loaded graphs and labels
    
    Returns:
        Tuple of (encoded_labels, label_mapping)
        - encoded_labels: numpy array of integer labels
        - label_mapping: dictionary mapping integer labels to original string labels
    """
    from sklearn.preprocessing import LabelEncoder
    
    if not loader.labels:
        # Labels not loaded - extract from graph paths
        loader.load_graphs()
        # Labels should now be populated from _load_edgelist
    
    if not loader.labels:
        raise ValueError("Could not extract labels from dataset")
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(loader.labels)
    
    # Create mapping: integer label -> original string label
    label_mapping = {
        i: label_name for i, label_name in enumerate(label_encoder.classes_)
    }
    
    return encoded_labels, label_mapping


def load_dgdvs_from_file(dgdv_file: str) -> List[np.ndarray]:
    """
    Load DGDVs from pickle file
    
    Args:
        dgdv_file: Path to DGDV pickle file (list of DGDV matrices)
    
    Returns:
        List of DGDV matrices
    """
    dgdv_path = Path(dgdv_file)
    
    if not dgdv_path.exists():
        raise FileNotFoundError(f"DGDV file not found: {dgdv_file}")
    
    with open(dgdv_path, 'rb') as f:
        dgdvs = pickle.load(f)
    
    if isinstance(dgdvs, list):
        return dgdvs
    else:
        raise ValueError(f"Unexpected DGDV file format. Expected list, got {type(dgdvs)}")


def load_dgdvs_from_individual_files(dgdv_dir: str, max_graphs: Optional[int] = None) -> List[np.ndarray]:
    """
    Load DGDVs from individual pickle files in a directory
    
    Args:
        dgdv_dir: Directory containing individual DGDV files (graph_XXXXX.pkl)
        max_graphs: Maximum number of graphs to load (None for all)
    
    Returns:
        List of DGDV matrices, sorted by graph index
    """
    from tqdm import tqdm
    
    dgdv_path = Path(dgdv_dir)
    
    if not dgdv_path.exists():
        raise FileNotFoundError(f"DGDV directory not found: {dgdv_dir}")
    
    # Find all graph_*.pkl files
    dgdv_files = sorted(dgdv_path.glob("graph_*.pkl"))
    
    if max_graphs:
        dgdv_files = dgdv_files[:max_graphs]
    
    print(f"Loading {len(dgdv_files)} DGDV files from: {dgdv_dir}")
    
    dgdvs = []
    for dgdv_file in tqdm(dgdv_files, desc="Loading DGDVs"):
        try:
            with open(dgdv_file, 'rb') as f:
                dgdv = pickle.load(f)
                dgdvs.append(dgdv)
        except Exception as e:
            print(f"Warning: Failed to load {dgdv_file}: {e}")
            # Add empty array to maintain alignment
            if dgdvs:
                # Use shape from previous DGDV
                dgdvs.append(np.zeros_like(dgdvs[0]))
            else:
                # First file failed - skip
                continue
    
    return dgdvs


def load_gcms_from_individual_files(gcm_dir: str, max_graphs: Optional[int] = None) -> np.ndarray:
    """
    Load GCMs from individual pickle files in a directory
    
    Args:
        gcm_dir: Directory containing individual GCM files (graph_XXXXX.pkl)
        max_graphs: Maximum number of graphs to load (None for all)
    
    Returns:
        Feature matrix of shape (n_graphs, n_gcm_features)
    """
    from tqdm import tqdm
    
    gcm_path = Path(gcm_dir)
    
    if not gcm_path.exists():
        raise FileNotFoundError(f"GCM directory not found: {gcm_dir}")
    
    # Find all graph_*.pkl files
    gcm_files = sorted(gcm_path.glob("graph_*.pkl"))
    
    if max_graphs:
        gcm_files = gcm_files[:max_graphs]
    
    print(f"Loading {len(gcm_files)} GCM files from: {gcm_dir}")
    
    gcms = []
    for gcm_file in tqdm(gcm_files, desc="Loading GCMs"):
        try:
            with open(gcm_file, 'rb') as f:
                gcm = pickle.load(f)
                gcms.append(gcm)
        except Exception as e:
            print(f"Warning: Failed to load {gcm_file}: {e}")
            # Will be handled in normalization step
    
    if not gcms:
        raise ValueError("No valid GCM vectors found")
    
    # Normalize GCMs to same size (same logic as load_gcm_features)
    valid_gcms = [gcm for gcm in gcms if gcm.size > 0]
    
    if not valid_gcms:
        raise ValueError("No valid GCM vectors found in files")
    
    # Find the maximum size
    max_size = max(gcm.size for gcm in valid_gcms)
    
    # Pad or truncate all GCMs to the same size
    normalized_gcms = []
    for gcm in gcms:
        if gcm.size == 0:
            # Empty GCM - pad with zeros
            normalized_gcm = np.zeros(max_size)
        elif gcm.size < max_size:
            # Pad with zeros
            normalized_gcm = np.pad(gcm, (0, max_size - gcm.size), mode='constant')
        elif gcm.size > max_size:
            # Truncate (shouldn't happen, but handle it)
            normalized_gcm = gcm[:max_size]
        else:
            normalized_gcm = gcm
        
        normalized_gcms.append(normalized_gcm)
    
    return np.array(normalized_gcms)


def load_features_from_files(
    dgdv_file: Optional[str] = None,
    gcm_file: Optional[str] = None,
    dgdv_dir: Optional[str] = None,
    gcm_dir: Optional[str] = None,
    data_dir: str = "data/malnet_tiny",
    max_graphs: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Dict, str]:
    """
    Load features and labels from files (supports both combined and individual files)
    
    Args:
        dgdv_file: Path to combined DGDV file (for coarse features) - mutually exclusive with dgdv_dir
        gcm_file: Path to combined GCM file - mutually exclusive with gcm_dir
        dgdv_dir: Directory with individual DGDV files (graph_XXXXX.pkl)
        gcm_dir: Directory with individual GCM files (graph_XXXXX.pkl)
        data_dir: Directory containing MalNet-Tiny dataset (for labels)
        max_graphs: Maximum number of graphs to load (None for all)
    
    Returns:
        Tuple of (features, labels, label_mapping, feature_type)
        - features: Feature matrix of shape (n_graphs, n_features)
        - labels: Encoded label array
        - label_mapping: Dictionary mapping integer labels to string labels
        - feature_type: "coarse", "gcm", or "both"
    """
    # Load labels
    loader = MalNetTinyLoader(data_dir)
    labels, label_mapping = load_labels_from_loader(loader)
    
    features_list = []
    feature_types = []
    
    # Load coarse features if requested
    if dgdv_dir:
        # Load from individual files
        dgdvs = load_dgdvs_from_individual_files(dgdv_dir, max_graphs=max_graphs)
        print(f"Extracting coarse features from {len(dgdvs)} DGDVs...")
        coarse_features = extract_coarse_features(dgdvs)
        features_list.append(coarse_features)
        feature_types.append("coarse")
        print(f"  Coarse features shape: {coarse_features.shape}")
    elif dgdv_file:
        # Load from combined file
        print(f"Loading DGDVs from: {dgdv_file}")
        dgdvs = load_dgdvs_from_file(dgdv_file)
        if max_graphs:
            dgdvs = dgdvs[:max_graphs]
        print(f"Extracting coarse features from {len(dgdvs)} DGDVs...")
        coarse_features = extract_coarse_features(dgdvs)
        features_list.append(coarse_features)
        feature_types.append("coarse")
        print(f"  Coarse features shape: {coarse_features.shape}")
    
    # Load GCM features if requested
    if gcm_dir:
        # Load from individual files
        gcm_features = load_gcms_from_individual_files(gcm_dir, max_graphs=max_graphs)
        features_list.append(gcm_features)
        feature_types.append("gcm")
        print(f"  GCM features shape: {gcm_features.shape}")
    elif gcm_file:
        # Load from combined file
        print(f"Loading GCMs from: {gcm_file}")
        gcm_features = load_gcm_features(gcm_file)
        if max_graphs:
            gcm_features = gcm_features[:max_graphs]
        features_list.append(gcm_features)
        feature_types.append("gcm")
        print(f"  GCM features shape: {gcm_features.shape}")
    
    if not features_list:
        raise ValueError("Must provide either dgdv_file/dgdv_dir or gcm_file/gcm_dir")
    
    # Combine features if both provided
    if len(features_list) == 2:
        # Ensure same number of graphs
        assert features_list[0].shape[0] == features_list[1].shape[0], \
            f"Feature count mismatch: {features_list[0].shape[0]} vs {features_list[1].shape[0]}"
        features = np.hstack(features_list)
        feature_type = "both"
    else:
        features = features_list[0]
        feature_type = feature_types[0]
    
    # Ensure labels match features
    if max_graphs:
        labels = labels[:max_graphs]
    
    if len(labels) != features.shape[0]:
        print(f"Warning: Label count ({len(labels)}) doesn't match feature count ({features.shape[0]})")
        min_count = min(len(labels), features.shape[0])
        labels = labels[:min_count]
        features = features[:min_count]
    
    return features, labels, label_mapping, feature_type

