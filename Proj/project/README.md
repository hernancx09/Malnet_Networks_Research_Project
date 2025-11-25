# MalNet-Tiny Graphlet Analysis

Compute Graphlet Degree Vectors (GDVs) for malware family classification using the MalNet-Tiny dataset.

## Quick Start

1. **Ensure ORCA is working** (64-bit Windows executable required)
2. **Run the pipeline**:
   ```bash
   python run_pipeline.py
   ```

## Project Structure

```
project/
├── src/              # Source code
│   ├── load_malnet.py          # Dataset loader
│   ├── orca_integration.py     # ORCA wrapper
│   └── compute_gdvs_malnet.py  # GDV computation pipeline
├── data/             # Dataset and results
│   ├── malnet_tiny/  # MalNet-Tiny dataset (5,000 graphs)
│   ├── gdvs/         # Computed GDV matrices
│   └── gcms/         # Computed GCM vectors
├── docs/             # Documentation
├── run_pipeline.py   # Main pipeline script
└── README.md         # This file
```

## Usage

### Load Dataset

```python
from src.load_malnet import MalNetTinyLoader

loader = MalNetTinyLoader("data/malnet_tiny")
graphs = loader.load_graphs()  # Loads 5,000 graphs
labels = loader.labels          # Family labels
```

### Compute GDVs

```python
from src.compute_gdvs_malnet import GDVProcessor

processor = GDVProcessor("data/malnet_tiny", "data/gdvs")
results = processor.process_graphs(graphs, max_graphlet_size=4)
```

### Run Full Pipeline

```bash
python run_pipeline.py
```

## Requirements

- Python 3.8+
- ORCA executable
- NetworkX, NumPy

Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

MalNet-Tiny contains 5,000 malware call graphs from 5 families:
- addisplay/kuguo
- adware/airpush
- benign/benign
- downloader/jiagu
- trojan/artemis

## Output

After running the pipeline:
- `data/gdvs/all_gdvs.pkl` - All GDV matrices (73 orbits per node)
- `data/gdvs/all_gcms.pkl` - All GCM vectors (graph-level features)
- `data/gdvs/metadata.json` - Processing metadata

## References

- Research Paper: "Classifying Malware Families Using Graphlet-Based Topological Signatures from MalNet-Tiny"
- ORCA Tool: https://www.biolab.si/supp/orca/