# Project Structure

## Directory Layout

```
project/
├── src/                    # Source code
│   ├── load_malnet.py          # Dataset loader
│   ├── orca_integration.py     # ORCA wrapper
│   ├── compute_gdvs_malnet.py  # GDV computation pipeline
│   └── README.md               # Source code documentation
│
├── data/                    # Data files
│   ├── malnet_tiny/          # MalNet-Tiny dataset (5,000 graphs)
│   ├── gdvs/                 # Computed GDV matrices (output)
│   ├── gcms/                 # Computed GCM vectors (output)
│   ├── processed/            # Processed graphs (intermediate)
│   ├── README.md             # Data directory documentation
│   └── GET_MALNET.md         # Dataset download instructions
│
├── docs/                     # Documentation
│   ├── Networks_Research (1).pdf  # Research paper
│   └── README.md             # Documentation index
│
├── orca.cpp                  # ORCA source code
├── orca.exe                  # ORCA executable
├── requirements.txt          # Python dependencies
├── run_pipeline.py           # Main pipeline script
├── run_pipeline.bat          # Windows batch file
└── README.md                 # Main project README
```

## Essential Files

**To run the project:**
- `run_pipeline.py` - Main entry point
- `src/` - All source code
- `data/malnet_tiny/` - Dataset (must be downloaded)
- `orca.exe` - ORCA executable (must be 64-bit compatible)

**Documentation:**
- `README.md` - Project overview and usage
- `data/GET_MALNET.md` - Dataset download instructions
- `docs/Networks_Research (1).pdf` - Research paper

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline
python run_pipeline.py
```

