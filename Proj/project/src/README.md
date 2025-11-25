# Source Code

## Files

- `load_malnet.py` - Loads MalNet-Tiny dataset from disk
- `orca_integration.py` - Wrapper for ORCA tool to compute GDVs
- `compute_gdvs_malnet.py` - Main pipeline for processing all graphs

## Usage

```python
# Load dataset
from load_malnet import MalNetTinyLoader
loader = MalNetTinyLoader("data/malnet_tiny")
graphs = loader.load_graphs()

# Compute GDVs
from compute_gdvs_malnet import GDVProcessor
processor = GDVProcessor("data/malnet_tiny", "data/gdvs")
results = processor.process_graphs(graphs, max_graphlet_size=4)
```

