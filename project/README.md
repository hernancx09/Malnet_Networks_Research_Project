# MalNet-Tiny Directed Graphlet Analysis

Compute Directed Graphlet Degree Vectors (DGDVs) and Graphlet Correlation Matrices (GCMs) for malware family classification using the MalNet-Tiny dataset and UCL Directed Graphlet Counter.

## Quick Start

1. **Ensure UCL Directed Graphlet Counter is compiled** (compile `Directed_Graphlet_Counter_v3.cpp`)
2. **Run the pipeline**:
   ```bash
   python run_pipeline.py
   ```

This will:
1. Load all MalNet-Tiny graphs (as directed graphs)
2. Compute DGDVs using UCL Directed Graphlet Counter
3. Apply orbit reduction (remove low-variance and collinear orbits)
4. Compute GCMs from reduced DGDVs
5. Save results to `data/gdvs/` (both combined and individual files)

## Requirements

- Python 3.8+
- WSL (Windows Subsystem for Linux) or Linux
- UCL Directed Graphlet Counter compiled (C++ compiler required)
- NetworkX, NumPy, tqdm

Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
project/
├── src/                          # Source code
│   ├── load_malnet.py                # Dataset loader
│   ├── ucl_integration.py            # UCL Directed Graphlet Counter wrapper
│   ├── compute_dgdvs_malnet.py     # DGDV computation pipeline
│   ├── orbit_reduction.py            # Orbit reduction module
│   ├── compute_gcms.py               # GCM computation module
│   ├── split_gcms.py                 # Utility to split GCM files
│   └── helpers/
│       └── gpu_acceleration.py       # GPU acceleration utilities
│
├── data/                        # Dataset and results
│   ├── malnet_tiny/              # MalNet-Tiny dataset (5,000 graphs)
│   └── gdvs/                     # Computed DGDVs and GCMs
│       ├── gdvs/                 # DGDV files
│       │   ├── all_dgdvs_3_4node.pkl          # Combined DGDVs
│       │   ├── all_dgdvs_3_4node_reduced.pkl # Reduced DGDVs
│       │   └── individual/                    # Individual DGDV files
│       │       └── graph_XXXXX.pkl
│       ├── gcms/                 # GCM files
│       │   ├── all_gcms_3_4node_reduced.pkl  # Combined GCMs
│       │   └── individual/                    # Individual GCM files
│       │       └── graph_XXXXX.pkl
│       └── metadata/              # Processing metadata
│
├── Directed_Graphlet_Counter_v3.cpp  # UCL counter source code
├── Directed_Graphlet_Counter_v3       # Compiled executable
│
├── run_pipeline.py              # Main pipeline script
├── run_pipeline.sh              # Shell script for cluster execution
└── README.md                    # This file
```

## Usage

### Run Full Pipeline

```bash
python run_pipeline.py
```

Or using the shell script:
```bash
./run_pipeline.sh
```

### Compile UCL Counter

First, compile the UCL Directed Graphlet Counter:
```bash
# On Linux/Mac:
g++ -O3 -o Directed_Graphlet_Counter_v3 Directed_Graphlet_Counter_v3.cpp

# On Windows (using WSL):
wsl g++ -O3 -o Directed_Graphlet_Counter_v3 Directed_Graphlet_Counter_v3.cpp
```

### Run Individual Steps

**1. Compute DGDVs:**
```bash
python run_pipeline.py
```

**2. Apply Orbit Reduction:**
```bash
python src/orbit_reduction.py \
    --input data/gdvs/gdvs/all_dgdvs_3_4node.pkl \
    --output data/gdvs/gdvs/all_dgdvs_3_4node_reduced.pkl
```

**3. Compute GCMs:**
```bash
python src/compute_gcms.py \
    --input data/gdvs/gdvs/all_dgdvs_3_4node_reduced.pkl \
    --output data/gdvs/gcms/all_gcms_3_4node_reduced.pkl
```

**4. Split Combined GCM File into Individual Files:**
```bash
python src/split_gcms.py \
    --input data/gdvs/gcms/all_gcms_3_4node_reduced.pkl
```

### Compute DGDVs Programmatically

```python
from src.load_malnet import MalNetTinyLoader
from src.ucl_integration import compute_directed_gdvs_with_ucl

# Load graphs
loader = MalNetTinyLoader("data/malnet_tiny")
graphs = loader.load_graphs(directed=True)

# Compute DGDV for a graph (3-node and 4-node graphlets)
dgdv = compute_directed_gdvs_with_ucl(graphs[0], min_graphlet_size=3, max_graphlet_size=4)
# Shape: (n_nodes, 127) for 3-node (39 orbits) + 4-node (88 orbits) directed motifs

# Or compute only 3-node:
dgdv_3node = compute_directed_gdvs_with_ucl(graphs[0], min_graphlet_size=3, max_graphlet_size=3)
# Shape: (n_nodes, 39) for 3-node directed motifs
```

### Orbit Reduction

Remove orbits with near-zero variance or strong collinearity:

```python
from src.orbit_reduction import reduce_orbits_from_file

reduce_orbits_from_file(
    input_file='data/gdvs/gdvs/all_dgdvs_3_4node.pkl',
    output_file='data/gdvs/gdvs/all_dgdvs_3_4node_reduced.pkl',
    variance_threshold=1e-6,      # Remove orbits with variance < 1e-6
    correlation_threshold=0.95    # Remove orbits with correlation > 0.95
)
```

### Compute GCMs

Compute Graphlet Correlation Matrices from DGDVs:

```python
from src.compute_gcms import compute_gcms_from_file

compute_gcms_from_file(
    input_file='data/gdvs/gdvs/all_dgdvs_3_4node_reduced.pkl',
    output_file='data/gdvs/gcms/all_gcms_3_4node_reduced.pkl',
    use_gpu=False,                # Set to True if GPU available
    save_individual=True           # Save individual GCM files
)
```

## Dataset

MalNet-Tiny contains 5,000 malware call graphs from 5 families:
- addisplay/kuguo
- adware/airpush
- benign/benign
- downloader/jiagu
- trojan/artemis

All edgelist files have been cleaned (headers removed) and are ready for processing.

## GPU Acceleration (Optional)

The pipeline supports GPU acceleration using **CuPy** for faster matrix operations:

- **2-10x speedup** for GCM computation on large graphs
- **Automatic fallback** to CPU if GPU unavailable

To use GPU acceleration:
```bash
python src/compute_gcms.py --input <dgdv_file> --output <gcm_file> --gpu
```

Quick setup:
```bash
# Check CUDA version: nvidia-smi
pip install cupy-cuda11x  # or cupy-cuda12x for CUDA 12.x
```

## Output

After running the pipeline:

**DGDV Files:**
- `data/gdvs/gdvs/all_dgdvs_3_4node.pkl` - Combined DGDV matrices (127 orbits)
- `data/gdvs/gdvs/all_dgdvs_3_4node_reduced.pkl` - Reduced DGDV matrices (after orbit reduction)
- `data/gdvs/gdvs/individual/graph_XXXXX.pkl` - Individual DGDV files (one per graph)

**GCM Files:**
- `data/gdvs/gcms/all_gcms_3_4node_reduced.pkl` - Combined GCM vectors
- `data/gdvs/gcms/individual/graph_XXXXX.pkl` - Individual GCM files (one per graph)

**Metadata:**
- `data/gdvs/metadata/processing_metadata_3_4node.json` - Processing metadata
- `data/gdvs/metadata/orbit_reduction_3_4node.json` - Orbit reduction statistics

### Loading Individual Files

```python
import pickle

# Load individual DGDV
with open('data/gdvs/gdvs/individual/graph_00042.pkl', 'rb') as f:
    dgdv = pickle.load(f)

# Load individual GCM
with open('data/gdvs/gcms/individual/graph_00042.pkl', 'rb') as f:
    gcm = pickle.load(f)
```

## DGDV Format

The UCL counter computes 129 orbits total:
- **2-node graphlets**: Orbits 0-1 (2 orbits)
- **3-node graphlets**: Orbits 2-40 (39 orbits) - 13 motifs × 3 orbits
- **4-node graphlets**: Orbits 41-128 (88 orbits)

**Default (3-node + 4-node)**:
- **Shape**: (n_nodes, 127) for 3-node + 4-node directed motifs
- **Columns**: 39 (3-node) + 88 (4-node) = 127 orbit positions

**3-node only**:
- **Shape**: (n_nodes, 39) for 3-node directed motifs
- **Columns**: 13 motifs × 3 orbits = 39 orbit positions

**Values**: Count of how many times each node appears in each orbit position

### DGDV Structure Example

```
DGDV Matrix (numpy.ndarray):
  Shape: (999, 127)  # 999 nodes × 127 orbits
  Dtype: int32
  Values: Non-negative integers (orbit counts)
  
  Structure:
    Row 0: [0, 3, 1, 42, 3, 1, ...]  # Node 0's orbit counts
    Row 1: [0, 2, 0, 38, 2, 0, ...]  # Node 1's orbit counts
    ...
    Row 998: [1, 5, 2, 51, 4, 2, ...]  # Node 998's orbit counts
    
  Interpretation:
    - Each row = one node
    - Each column = one orbit
    - Value = count of how many times that node appears in that orbit
```

### GCM Structure Example

```
GCM Vector (numpy.ndarray):
  Shape: (325,)  # Flattened upper triangle
  Dtype: float64
  Values: Correlation coefficients in range [-1, 1]
  
  Structure:
    [0.545, 0.014, 0.051, 0.437, 0.019, ...]  # 325 correlation values
    
  Interpretation:
    - Represents upper triangle of correlation matrix (excluding diagonal)
    - Original matrix: 26 × 26 orbits (after orbit reduction)
    - Formula: n × (n-1) / 2 = 26 × 25 / 2 = 325 correlations
    - Each value = correlation between two orbits (across nodes)
    
  Matrix Reconstruction:
    corr_matrix = np.zeros((26, 26))
    triu_indices = np.triu_indices(26, k=1)
    corr_matrix[triu_indices] = gcm
    corr_matrix.T[triu_indices] = gcm  # Make symmetric
    np.fill_diagonal(corr_matrix, 1.0)  # Self-correlation
```

### Format Comparison

| Feature | DGDV | GCM |
|---------|------|-----|
| **Level** | Node-level | Graph-level |
| **Structure** | Matrix (n_nodes × n_orbits) | Vector (n_correlations,) |
| **Example Shape** | (999, 127) | (325,) |
| **Data Type** | int32 | float64 |
| **Values** | Non-negative integers (counts) | Floats in [-1, 1] (correlations) |
| **Meaning** | Orbit counts per node | Orbit pair correlations |
| **Computation** | Direct from UCL counter | Computed from DGDV |

**Relationship**: GCM is computed FROM DGDV by:
1. Starting with DGDV: (n_nodes, n_orbits) matrix
2. Computing correlation between orbits (across nodes)
3. Resulting in (n_orbits, n_orbits) correlation matrix
4. Extracting upper triangle (excluding diagonal)
5. Flattening to vector: n_orbits × (n_orbits-1) / 2 elements

## GCM Format

Graphlet Correlation Matrices (GCMs) are graph-level features computed from DGDVs:

1. **Correlation Matrix**: For each graph, compute correlation between orbits (across nodes)
   - Input: DGDV matrix of shape (n_nodes, n_orbits)
   - Output: Correlation matrix of shape (n_orbits, n_orbits)

2. **Flattened Vector**: Extract upper triangular portion (excluding diagonal)
   - Size: n_orbits × (n_orbits - 1) / 2
   - Contains pairwise correlations between all orbit pairs

3. **After Orbit Reduction**: 
   - GCM size depends on number of orbits kept after reduction
   - Typically 200-700 elements for reduced 3-4 node graphlets

**Size Examples**:
- Original: 127 orbits → GCM size = 127 × 126 / 2 = 8,001 elements
- Reduced: ~26 orbits → GCM size = 26 × 25 / 2 = 325 elements
- Reduced: ~50 orbits → GCM size = 50 × 49 / 2 = 1,225 elements

## Memory Management

The pipeline uses incremental saving to handle large datasets:

- **Individual files**: Each DGDV/GCM saved immediately after computation
- **Checkpointing**: Progress saved periodically to prevent data loss
- **Resume capability**: Can resume from last checkpoint if interrupted
- **Batch processing**: Large operations split into batches

## References

- Research Paper: "Classifying Malware Families Using Graphlet-Based Topological Signatures from MalNet-Tiny"
- UCL Directed Graphlet Counter: http://www0.cs.ucl.ac.uk/staff/natasa/DGCD/index.html
- Dataset: https://github.com/nd7141/graph_datasets
