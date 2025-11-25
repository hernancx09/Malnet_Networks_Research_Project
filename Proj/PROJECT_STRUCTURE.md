# Project Structure Summary

## Files Organized

Your project has been organized into the following structure:

```
Proj/
├── demos/                          # Demo and example files
│   ├── orca_demo.py                # Network topology demos
│   ├── star_network.png
│   ├── mesh_network.png
│   ├── ring_network.png
│   └── enterprise_network.png
│
├── project/                        # MalNet-Tiny
│   ├── src/                       
│   │   ├── orca_integration.py     # ORCA wrapper
│   │   ├── orca_python_fallback.py # Python graphlet counter
│   │   ├── load_malnet.py          # Dataset loader
│   │   └── compute_gdvs_malnet.py  # GDV computation pipeline
│   │
│   ├── data/                       # Dataset files
│   │   ├── malnet_tiny/            # MalNet-Tiny dataset (download required)
│   │   ├── processed/              # Processed graphs
│   │   ├── gdvs/                   # Computed GDV matrices
│   │   └── gcms/                   # Computed GCM vectors
│   │
│   ├── docs/                       # Documentation
│   │   ├── Networks_Research (1).pdf
│   │   ├── README.md
│   │   └── ...
│   │
│   ├── orca.exe                    # ORCA executable
│   ├── orca.cpp                    # ORCA source code
│   ├── requirements.txt            # Python dependencies
│   ├── README.md                   # Project README
│   ├── run_pipeline.py             # Main pipeline script
│   ├── download_malnet.py          # Dataset download helper
│   └── GET_MALNET.md               # Download instructions
│
├── PROJECT_STRUCTURE.md            # You get the idea
└── run_pipeline.sh                 # script running on CRC
```

## Next Steps

### 1. Download MalNet-Tiny Dataset

### 2. Run the Pipeline

Once dataset is downloaded:

```bash
cd project
python run_pipeline.py
```

This will:
- Load all graphs from MalNet-Tiny
- Compute GDVs for 2-, 3-, and 4-node graphlets
- Compute GCMs (Graphlet Correlation Matrices)
- Save results to `project/data/gdvs/`

### 3. Use Results for Classification

The computed GDVs and GCMs can be loaded and used for malware family classification:

## What Each Directory Contains

### `demos/`
- Example network topologies
- Visualization demos
- Learning examples

### `project/src/`
- **load_malnet.py** - Loads MalNet-Tiny dataset from disk
- **orca_integration.py** - Wrapper for ORCA tool to compute GDVs
- **compute_gdvs_malnet.py** - Main pipeline for processing all graphs

### `project/data/`
- **malnet_tiny/**: Dataset files (download required)
- **gdvs/**: Computed Graphlet Degree Vectors
- **gcms/**: Computed Graphlet Correlation Matrices
- **processed/**: Intermediate processed graphs

### `project/docs/`
- Research paper
- Documentation

## Project Goals

As described in research paper:

1. **Extract graphlet features** using ORCA (2-, 3-, 4-node graphlets)
2. **Compute GDVs** (Graphlet Degree Vectors) for each graph
3. **Compute GCMs** (Graphlet Correlation Matrices) for graph-level features
4. **Classify malware families** using graphlet-based features




