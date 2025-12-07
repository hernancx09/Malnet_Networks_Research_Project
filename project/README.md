# MalNet-Tiny Directed Graphlet Analysis

Compute Directed Graphlet Degree Vectors (DGDVs) for malware family classification using the MalNet-Tiny dataset and UCL Directed Graphlet Counter.

## Quick Start

1. **Ensure UCL Directed Graphlet Counter is compiled** (compile `Directed_Graphlet_Counter_v3.cpp`)
2. **Run the pipeline**:
   ```bash
   python run_pipeline.py
   ```

## Requirements

- Python 3.8+
- WSL (Windows Subsystem for Linux) or Linux
- UCL Directed Graphlet Counter compiled (C++ compiler required)
- NetworkX, NumPy

Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
project/
├── src/                    # Source code
│   ├── load_malnet.py          # Dataset loader
│   ├── ucl_integration.py      # UCL Directed Graphlet Counter wrapper
│   ├── directed_motifs.py     # Motif classification
│   ├── compute_dgdvs_malnet.py # DGDV computation pipeline
│   └── clean_edgelists.py     # File cleaning utility
│
├── data/                    # Dataset and results
│   ├── malnet_tiny/          # MalNet-Tiny dataset (5,000 graphs)
│   ├── gdvs/                 # Computed DGDV matrices
│   └── gcms/                 # Computed GCM vectors
│
├── Directed_Graphlet_Counter_v3.cpp  # UCL counter source code
├── Directed_Graphlet_Counter_v3       # Compiled executable
│
├── run_pipeline.py          # Main pipeline script
└── README.md                # This file
```

## Usage

### Run Full Pipeline

```bash
python run_pipeline.py
```

This will:
1. Load all MalNet-Tiny graphs (as directed graphs)
2. Compute DGDVs using UCL Directed Graphlet Counter
3. Compute GCMs from DGDVs
4. Save results to `data/gdvs/`

### Compile UCL Counter

First, compile the UCL Directed Graphlet Counter:
```bash
# On Linux/Mac:
g++ -O3 -o Directed_Graphlet_Counter_v3 Directed_Graphlet_Counter_v3.cpp

# On Windows (using WSL):
wsl g++ -O3 -o Directed_Graphlet_Counter_v3 Directed_Graphlet_Counter_v3.cpp
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

### Process Subset

```python
from src.compute_dgdvs_malnet import DGDVProcessor

processor = DGDVProcessor("data/malnet_tiny", "data/gdvs")
results = processor.process_graphs(graphs[:100], limit=100)  # Process first 100
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
- **1.5-3x speedup** for DGDV matrix construction
- **Automatic fallback** to CPU if GPU unavailable

See [GPU_SETUP.md](GPU_SETUP.md) for installation and usage instructions.

Quick setup:
```bash
# Check CUDA version: nvidia-smi
pip install cupy-cuda11x  # or cupy-cuda12x for CUDA 12.x
```

## Output

After running the pipeline:
- `data/gdvs/all_dgdvs.pkl` - All DGDV matrices (39 orbits per node for 3-node motifs)
- `data/gdvs/all_gcms.pkl` - All GCM vectors (graph-level features)
- `data/gdvs/metadata/processing_metadata.json` - Processing metadata

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

## References

- Research Paper: "Classifying Malware Families Using Graphlet-Based Topological Signatures from MalNet-Tiny"
- UCL Directed Graphlet Counter: http://www0.cs.ucl.ac.uk/staff/natasa/DGCD/index.html
- Dataset: https://github.com/nd7141/graph_datasets
