#!/usr/bin/env python3
"""
UCL Directed Graphlet Counter Integration
Computes Directed Graphlet Degree Vectors (DGDVs) using UCL Directed Graphlet Counter
Based on: http://www0.cs.ucl.ac.uk/staff/natasa/DGCD/index.html
"""

import subprocess
import numpy as np
import networkx as nx
import os
import tempfile
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import platform

def compute_directed_gdvs_with_ucl(graph: nx.DiGraph,
                                   ucl_exe_path: Optional[str] = None,
                                   min_graphlet_size: int = 3,
                                   max_graphlet_size: int = 3) -> Optional[np.ndarray]:
    """
    Compute Directed Graphlet Degree Vectors (DGDVs) using UCL Directed Graphlet Counter
    
    UCL counter outputs 129 orbits total:
    - Orbits 0-1: 2-node graphlets (2 orbits)
    - Orbits 2-40: 3-node graphlets (39 orbits)
    - Orbits 41-128: 4-node graphlets (88 orbits)
    
    Args:
        graph: Directed NetworkX graph object
        ucl_exe_path: Path to UCL Directed_Graphlet_Counter_v3 executable
        min_graphlet_size: Minimum graphlet size to extract (2, 3, or 4)
        max_graphlet_size: Maximum graphlet size to extract (2, 3, or 4)
    
    Returns:
        DGDV matrix where each row is a node and columns are orbit counts,
        or None if UCL counter is not available
    """
    # Find UCL executable
    if ucl_exe_path is None:
        # Try local executable first
        local_ucl = os.path.join(os.path.dirname(__file__), '..', 'Directed_Graphlet_Counter_v3')
        if os.path.exists(local_ucl):
            ucl_exe_path = local_ucl
        else:
            ucl_exe_path = 'Directed_Graphlet_Counter_v3'  # Try system PATH
    
    if not os.path.exists(ucl_exe_path):
        print(f"UCL Directed Graphlet Counter not found at '{ucl_exe_path}'")
        return None
    
    # Convert graph to UCL format (edge list, space-separated, 1-indexed)
    edge_file_path = _graph_to_ucl_format(graph)
    
    try:
        # Run UCL counter
        result = _run_ucl_counter(ucl_exe_path, edge_file_path)
        
        if result is None:
            return None
        
        signatures_file, dictionary_file = result
        
        # Parse signatures
        dgdv = _parse_ucl_signatures(signatures_file, dictionary_file, graph, min_graphlet_size, max_graphlet_size)
        
        return dgdv
        
    except Exception as e:
        print(f"Error running UCL counter: {e}")
        return None
    finally:
        # Clean up temporary edge file
        try:
            os.unlink(edge_file_path)
        except:
            pass


def _graph_to_ucl_format(graph: nx.DiGraph) -> str:
    """
    Convert NetworkX DiGraph to UCL format
    
    Format: Simple edge list, space-separated, 1-indexed nodes
    "a b" means edge from a to b
    """
    # Create 1-indexed node mapping
    nodes = sorted(graph.nodes())
    node_mapping = {node: i+1 for i, node in enumerate(nodes)}
    
    # Write edges to temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.edgelist', delete=False)
    
    for u, v in graph.edges():
        u_idx = node_mapping[u]
        v_idx = node_mapping[v]
        # Skip self-loops (UCL counter removes them anyway)
        if u_idx != v_idx:
            temp_file.write(f"{u_idx} {v_idx}\n")
    
    temp_file.close()
    return temp_file.name


def _run_ucl_counter(exe_path: str, edge_file: str) -> Optional[Tuple[str, str]]:
    """
    Run UCL Directed Graphlet Counter on the graph file
    
    Returns:
        Tuple of (signatures_file, dictionary_file) paths, or None on failure
    """
    import platform
    
    # UCL counter creates output files based on input filename
    # Output: input_file.signatures.txt, input_file.dictionary.txt, input_file.graphletcounts.txt
    edge_file_base = os.path.basename(edge_file)
    edge_file_dir = os.path.dirname(edge_file)
    
    signatures_file = os.path.join(edge_file_dir, f"{edge_file_base}.signatures.txt")
    dictionary_file = os.path.join(edge_file_dir, f"{edge_file_base}.dictionary.txt")
    
    if platform.system() == "Windows":
        # Convert to WSL paths
        def to_wsl_path(p):
            p_str = str(p).replace('\\', '/')
            if p_str[1] == ':':
                drive = p_str[0].lower()
                return f"/mnt/{drive}{p_str[2:]}"
            return p_str
        
        exe_wsl = to_wsl_path(Path(exe_path).absolute())
        edge_wsl = to_wsl_path(Path(edge_file).absolute())
        edge_dir = os.path.dirname(edge_wsl)
        edge_base = os.path.basename(edge_wsl)
        
        cmd = ["wsl", "bash", "-c", f"cd '{edge_dir}' && '{exe_wsl}' '{edge_base}'"]
    else:
        # On Linux/Mac, run directly
        edge_dir = os.path.dirname(edge_file)
        edge_base = os.path.basename(edge_file)
        cmd = [exe_path, edge_base]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=edge_dir if platform.system() != "Windows" else None
        )
        
        if result.returncode != 0:
            print(f"UCL Counter failed: {result.stderr}")
            return None
        
        # Wait a moment for file system sync (especially on Windows/WSL)
        import time
        time.sleep(1)
        
        # Check if output files exist
        if not os.path.exists(signatures_file):
            print(f"Signatures file not found: {signatures_file}")
            return None
        
        return signatures_file, dictionary_file
        
    except subprocess.TimeoutExpired:
        print("UCL Counter timed out")
        return None
    except Exception as e:
        print(f"Error running UCL Counter: {e}")
        return None


def _parse_ucl_signatures(signatures_file: str,
                         dictionary_file: str,
                         graph: nx.DiGraph,
                         min_graphlet_size: int = 3,
                         max_graphlet_size: int = 3) -> Optional[np.ndarray]:
    """
    Parse UCL signatures file and build DGDV matrix
    
    UCL counter outputs:
    - signatures.txt: Each line is a space-separated list of 129 orbit counts (0-128)
                     Orbits 0-1: 2-node graphlets (2 orbits)
                     Orbits 2-40: 3-node graphlets (39 orbits)
                     Orbits 41-128: 4-node graphlets (88 orbits)
    - dictionary.txt: Maps line number to original node ID
    
    Args:
        signatures_file: Path to signatures.txt
        dictionary_file: Path to dictionary.txt
        graph: Original NetworkX graph (for node ordering)
        min_graphlet_size: Minimum graphlet size to extract (2, 3, or 4)
        max_graphlet_size: Maximum graphlet size to extract (2, 3, or 4)
    
    Returns:
        DGDV matrix (nodes × orbits)
    """
    # Read signatures
    signatures = []
    with open(signatures_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Each line is a space-separated list of orbit counts
            counts = [int(x) for x in line.split()]
            signatures.append(counts)
    
    if not signatures:
        print("No signatures found in file")
        return None
    
    # Convert to numpy array
    signatures_array = np.array(signatures)
    
    # Determine orbit ranges based on graphlet sizes
    # UCL counter orbit ranges:
    # - 2-node: orbits 0-1 (2 orbits)
    # - 3-node: orbits 2-40 (39 orbits)
    # - 4-node: orbits 41-128 (88 orbits)
    
    if signatures_array.shape[1] < 129:
        print(f"Warning: Expected 129 orbits, got {signatures_array.shape[1]}")
        # Return what we have if incomplete
        return signatures_array
    
    # Extract orbits based on requested graphlet sizes
    orbit_ranges = []
    
    if min_graphlet_size <= 2 <= max_graphlet_size:
        orbit_ranges.append((0, 2))  # 2-node: orbits 0-1
    
    if min_graphlet_size <= 3 <= max_graphlet_size:
        orbit_ranges.append((2, 41))  # 3-node: orbits 2-40
    
    if min_graphlet_size <= 4 <= max_graphlet_size:
        orbit_ranges.append((41, 129))  # 4-node: orbits 41-128
    
    if not orbit_ranges:
        print(f"Warning: Invalid graphlet size range {min_graphlet_size}-{max_graphlet_size}")
        return None
    
    # Concatenate selected orbit ranges
    selected_orbits = []
    for start, end in orbit_ranges:
        selected_orbits.append(signatures_array[:, start:end])
    
    if len(selected_orbits) == 1:
        dgdv = selected_orbits[0]
    else:
        dgdv = np.hstack(selected_orbits)
    
    # Verify node count matches
    n_nodes_graph = graph.number_of_nodes()
    n_nodes_signatures = dgdv.shape[0]
    
    if n_nodes_graph != n_nodes_signatures:
        print(f"Warning: Graph has {n_nodes_graph} nodes but signatures have {n_nodes_signatures} nodes")
        # UCL counter may exclude nodes that only have self-loops
    
    return dgdv


def check_ucl_available(ucl_exe_path: Optional[str] = None) -> bool:
    """Check if UCL Directed Graphlet Counter is available"""
    if ucl_exe_path is None:
        local_ucl = os.path.join(os.path.dirname(__file__), '..', 'Directed_Graphlet_Counter_v3')
        if os.path.exists(local_ucl):
            return True
        return False
    return os.path.exists(ucl_exe_path)


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("UCL Directed Graphlet Counter Integration Test")
    print("="*60)
    
    if not check_ucl_available():
        print("\n[WARNING] UCL Directed Graphlet Counter is not available.")
        print("Please compile Directed_Graphlet_Counter_v3.cpp first.")
    else:
        print("\n[SUCCESS] UCL Directed Graphlet Counter is available!\n")
    
    # Create a small test directed graph
    print("Creating test directed graph...")
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (0, 2)])  # Transitive triangle
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Try to compute DGDVs (3-node and 4-node)
    print("\nComputing Directed Graphlet Degree Vectors...")
    dgdvs = compute_directed_gdvs_with_ucl(G, min_graphlet_size=3, max_graphlet_size=4)
    
    if dgdvs is not None:
        print(f"[SUCCESS] DGDV matrix computed!")
        print(f"  Shape: {dgdvs.shape} (nodes × orbits)")
        print(f"  First node's orbit counts (first 10): {dgdvs[0][:10]}")
        print(f"  Total orbits: {dgdvs.shape[1]} (39 for 3-node + 88 for 4-node = 127)")
    else:
        print("\n[INFO] DGDV computation failed")
    
    print("\n" + "="*60)

