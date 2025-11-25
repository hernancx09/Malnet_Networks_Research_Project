#!/usr/bin/env python3
"""
ORCA Integration for Network Models
Computes Graphlet Degree Vectors (GDVs) using ORCA
"""

import subprocess
import numpy as np
import networkx as nx
import os
import tempfile
from typing import Optional, Tuple

def compute_gdvs_with_orca(graph: nx.Graph, max_graphlet_size: int = 4, 
                           orca_path: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Compute Graphlet Degree Vectors using ORCA
    
    Args:
        graph: NetworkX graph object
        max_graphlet_size: Maximum graphlet size (2, 3, or 4)
        orca_path: Path to ORCA executable (default: 'orca' in PATH)
    
    Returns:
        GDV matrix where each row is a node and columns are orbit counts,
        or None if ORCA is not available
    """
    # Try to find ORCA executable
    if orca_path is None:
        # Try local orca.exe first
        local_orca = os.path.join(os.path.dirname(__file__), 'orca.exe')
        if os.path.exists(local_orca):
            orca_path = local_orca
        else:
            orca_path = 'orca'  # Try system PATH
    
    # Convert graph to edge list format
    # ORCA expects node IDs to be integers starting from 0
    node_mapping = {node: i for i, node in enumerate(graph.nodes())}
    reverse_mapping = {i: node for node, i in node_mapping.items()}
    
    n = len(node_mapping)  # Number of nodes
    m = graph.number_of_edges()  # Number of edges
    
    # ORCA expects: first line = "n m", then edges
    edge_list = [f"{n} {m}\n"]
    for u, v in graph.edges():
        u_idx = node_mapping[u]
        v_idx = node_mapping[v]
        edge_list.append(f"{u_idx} {v_idx}\n")
    
    # Use temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as edge_file:
        edge_file.writelines(edge_list)
        edge_file_path = edge_file.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as output_file:
        output_file_path = output_file.name
    
    try:
        # Run ORCA - use shell=True on Windows if needed
        cmd = [orca_path, str(max_graphlet_size), edge_file_path, output_file_path]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300  # 5 minute timeout
        )
        
        # Read orbit counts
        # ORCA outputs binary format, but we can try text first
        orbits = []
        try:
            # Try reading as text first
            with open(output_file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # ORCA outputs space-separated orbit counts
                        orbit_counts = list(map(int, line.split()))
                        orbits.append(orbit_counts)
        except:
            # If text fails, try binary (though ORCA uses fstream::binary)
            # Actually, ORCA writes text even with binary flag, so this should work
            pass
        
        if orbits:
            return np.array(orbits)
        else:
            print("Warning: ORCA produced no output")
            return None
    
    except FileNotFoundError:
        print(f"ORCA executable not found at '{orca_path}'.")
        print("Please install ORCA. See misc/compile_orca_instructions.md for compilation instructions.")
        raise
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"ORCA execution error: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"ORCA stderr: {e.stderr}")
        if hasattr(e, 'stdout') and e.stdout:
            print(f"ORCA stdout: {e.stdout}")
        raise
    except subprocess.TimeoutExpired:
        print("ORCA computation timed out (graph may be too large)")
        raise
    finally:
        # Clean up temporary files
        try:
            os.unlink(edge_file_path)
            os.unlink(output_file_path)
        except:
            pass


def compute_gcm(gdvs: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute Graphlet Correlation Matrix (GCM) as described in the research paper.
    
    The GCM captures relationships between orbit counts across nodes.
    The upper triangular portion is flattened into a feature vector.
    
    Args:
        gdvs: Graphlet Degree Vector matrix (nodes × orbits)
    
    Returns:
        Flattened GCM vector (upper triangular portion of correlation matrix)
    """
    if gdvs is None or gdvs.size == 0:
        return None
    
    # Compute correlation matrix between orbits (columns)
    # This shows how different orbit types correlate across nodes
    correlation_matrix = np.corrcoef(gdvs.T)
    
    # Handle NaN values (can occur if an orbit has zero variance)
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
    
    # Extract upper triangular portion (excluding diagonal)
    # This gives us a compact representation of orbit relationships
    n_orbits = correlation_matrix.shape[0]
    upper_tri_indices = np.triu_indices(n_orbits, k=1)
    gcm_vector = correlation_matrix[upper_tri_indices]
    
    return gcm_vector


def check_orca_available(orca_path: Optional[str] = None) -> bool:
    """
    Check if ORCA is available in the system
    
    Args:
        orca_path: Path to ORCA executable
    
    Returns:
        True if ORCA is available, False otherwise
    """
    if orca_path is None:
        # Try local orca.exe first
        local_orca = os.path.join(os.path.dirname(__file__), 'orca.exe')
        if os.path.exists(local_orca):
            orca_path = local_orca
        else:
            orca_path = 'orca'  # Try system PATH
    
    # Check if file exists
    if os.path.exists(orca_path):
        return True
    
    # Try to run it
    try:
        result = subprocess.run(
            [orca_path, '--help'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0 or 'orca' in result.stderr.lower() or 'orca' in result.stdout.lower()
    except:
        return False


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("ORCA Integration Test")
    print("="*60)
    
    # Check if ORCA is available
    if not check_orca_available():
        print("\n[WARNING] ORCA is not available in your system.")
        print("Please install ORCA to use graphlet features.")
        print("See ORCA_SETUP.md for installation instructions.\n")
        print("Creating a sample graph for demonstration...")
    else:
        print("\n[SUCCESS] ORCA is available!\n")
    
    # Create a sample graph
    print("Creating sample graph (Karate Club)...")
    G = nx.karate_club_graph()
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Try to compute GDVs
    print("\nComputing Graphlet Degree Vectors...")
    gdvs = compute_gdvs_with_orca(G, max_graphlet_size=4)
    
    if gdvs is not None:
        print(f"[SUCCESS] GDV matrix computed!")
        print(f"  Shape: {gdvs.shape} (nodes × orbits)")
        print(f"  Number of orbits: {gdvs.shape[1]}")
        print(f"  First node's orbit counts (first 10): {gdvs[0][:10]}")
        
        # Compute GCM
        print("\nComputing Graphlet Correlation Matrix (GCM)...")
        gcm = compute_gcm(gdvs)
        if gcm is not None:
            print(f"[SUCCESS] GCM vector computed!")
            print(f"  GCM vector length: {len(gcm)}")
            print(f"  First 10 GCM values: {gcm[:10]}")
    else:
        print("\n[INFO] GDV computation skipped (ORCA not available)")
        print("This is expected if ORCA is not installed.")
        print("\nTo use ORCA:")
        print("1. Install ORCA (see ORCA_SETUP.md)")
        print("2. Ensure 'orca' is in your PATH")
        print("3. Re-run this script")
    
    print("\n" + "="*60)

