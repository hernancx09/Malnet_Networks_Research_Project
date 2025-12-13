# MalNet-Tiny Directed Graphlet Analysis

Compute Directed Graphlet Degree Vectors (DGDVs) and Graphlet Correlation Matrices (GCMs) for malware family classification using the MalNet-Tiny dataset and UCL Directed Graphlet Counter.

## Quick Start

### Complete Pipeline Workflow

Follow these steps to run the complete pipeline from feature extraction to classification:

#### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 2: Compile UCL Directed Graphlet Counter

```bash
# On Linux/Mac:
g++ -O3 -o Directed_Graphlet_Counter_v3 Directed_Graphlet_Counter_v3.cpp

# On Windows (using WSL):
wsl g++ -O3 -o Directed_Graphlet_Counter_v3 Directed_Graphlet_Counter_v3.cpp
```

#### Step 3: Compute DGDVs

This will load all graphs, compute DGDVs, apply orbit reduction, and compute GCMs:

```bash
python run_pipeline.py
```

**What this does:**
- Loads all 5,000 MalNet-Tiny graphs (as directed graphs)
- Computes DGDVs using UCL Directed Graphlet Counter (3-4 node graphlets, 127 orbits)
- Applies orbit reduction (removes low-variance and collinear orbits)
- Computes GCMs from reduced DGDVs
- Saves individual files to:
  - `data/DGDVs/individual/graph_XXXXX.pkl` (one DGDV per graph)
  - `data/GCMs/individual/graph_XXXXX.pkl` (one GCM per graph)

**Expected output:**
- 5,000 individual DGDV files in `data/DGDVs/individual/`
- 5,000 individual GCM files in `data/GCMs/individual/`
- Metadata files in `data/DGDVs/metadata/`

#### Step 4: Run Classification

After features are computed, run classification:

```bash
# Classify using GCM features with both classifiers
python src/models/run_classification.py --features gcm --classifier all --split train_test

# Or use coarse features from DGDVs
python src/models/run_classification.py --features coarse --classifier all --split train_test

# Or use both feature types
python src/models/run_classification.py --features both --classifier all --split train_test
```

**What this does:**
- Loads features from individual files (default: `data/GCMs/individual/` or `data/DGDVs/individual/`)
- Loads labels from dataset directory
- Splits data: 60% train, 20% validation, 20% test (stratified)
- Trains Random Forest and SVM classifiers
- Evaluates on test set
- Saves models, metrics, and confusion matrices to `data/results/`

**Expected output:**
- Trained models: `data/results/models/rf_gcm_model.pkl`, `svm_gcm_model.pkl`
- Evaluation results: `data/results/classification_results/*.json` and `*.csv`
- Confusion matrices: `data/results/classification_results/confusion_matrices/*.csv` and `*.png`
- Comparison report: `data/results/classification_results/comparison_report.txt`

### Quick Test (Small Subset)

To test the pipeline on a small subset first:

```bash
# Compute features for first 100 graphs only (modify run_pipeline.py or use subset)
# Then test classification
python src/models/run_classification.py --features gcm --classifier rf --split train_test
```

## Requirements

- Python 3.8+
- WSL (Windows Subsystem for Linux) or Linux
- UCL Directed Graphlet Counter compiled (C++ compiler required)
- NetworkX, NumPy, tqdm
- scikit-learn, pandas, matplotlib (for classification)

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
│   ├── models/                       # Classification models
│   │   ├── classification.py         # Classification pipeline (RF, SVM)
│   │   ├── feature_extraction.py     # Feature extraction from DGDVs/GCMs
│   │   ├── evaluation.py             # Evaluation metrics and reporting
│   │   └── run_classification.py     # Main classification script
│   └── helpers/
│       └── gpu_acceleration.py       # GPU acceleration utilities
│
├── data/                        # Dataset and results
│   ├── malnet_tiny/              # MalNet-Tiny dataset (5,000 graphs)
│   ├── DGDVs/                    # Computed DGDVs
│   │   ├── individual/           # Individual DGDV files (graph_XXXXX.pkl)
│   │   └── metadata/             # Processing metadata
│   ├── GCMs/                     # Computed GCMs
│   │   └── individual/           # Individual GCM files (graph_XXXXX.pkl)
│   └── results/                  # Classification results
│       ├── models/               # Trained classifier models
│       └── classification_results/  # Evaluation results and reports
│
├── Directed_Graphlet_Counter_v3.cpp  # UCL counter source code
├── Directed_Graphlet_Counter_v3       # Compiled executable
│
├── run_pipeline.py              # Main pipeline script
├── run_pipeline.sh              # Shell script for cluster execution
└── README.md                    # This file
```

## Complete Workflow Guide

This section provides a step-by-step guide to run the entire pipeline correctly from start to finish.

### Prerequisites Checklist

Before starting, ensure you have:

- [ ] Python 3.8+ installed
- [ ] All dependencies installed: `pip install -r requirements.txt`
- [ ] UCL Directed Graphlet Counter compiled (see below)
- [ ] MalNet-Tiny dataset in `data/malnet_tiny/` directory
- [ ] At least 20GB free disk space (for 5,000 graphs)

### Step-by-Step Execution

#### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd Malnet_Networks_Research_Project/project

# Install Python packages
pip install -r requirements.txt
```

**Verify installation:**
```bash
python -c "import networkx, numpy, sklearn, pandas, matplotlib; print('All dependencies OK!')"
```

#### Step 2: Compile UCL Counter

```bash
# On Linux/Mac:
g++ -O3 -o Directed_Graphlet_Counter_v3 Directed_Graphlet_Counter_v3.cpp

# On Windows (using WSL):
wsl g++ -O3 -o Directed_Graphlet_Counter_v3 Directed_Graphlet_Counter_v3.cpp
```

**Verify compilation:**
```bash
# Should show executable file
ls -lh Directed_Graphlet_Counter_v3
# Or on Windows:
dir Directed_Graphlet_Counter_v3
```

#### Step 3: Verify Dataset

```bash
# Check that dataset exists
ls data/malnet_tiny/malnet-graphs-tiny/
# Should show 5 directories: addisplay, adware, benign, downloader, trojan
```

#### Step 4: Compute Features (DGDVs and GCMs)

```bash
python run_pipeline.py
```

**What happens:**
1. Loads all 5,000 graphs from `data/malnet_tiny/`
2. Computes DGDVs for each graph (3-4 node graphlets, 127 orbits)
3. Saves individual DGDV files to `data/DGDVs/individual/graph_XXXXX.pkl`
4. Applies orbit reduction (removes low-variance/collinear orbits)
5. Computes GCMs from reduced DGDVs
6. Saves individual GCM files to `data/GCMs/individual/graph_XXXXX.pkl`

**Expected output:**
- Progress bars showing computation progress
- Final message: "Pipeline completed successfully!"
- 5,000 files in `data/DGDVs/individual/`
- 5,000 files in `data/GCMs/individual/`

**Time estimate:** 2-6 hours depending on hardware (CPU/GPU)

**Troubleshooting:**
- If out of memory: The pipeline uses incremental saving, but you may need to process in batches
- If interrupted: The pipeline can resume from checkpoints (check metadata files)
- If UCL counter fails: Verify executable is compiled and has execute permissions

#### Step 5: Run Classification

After features are computed, run classification:

```bash
# Option 1: GCM features with both classifiers (recommended)
python src/models/run_classification.py --features gcm --classifier all --split train_test

# Option 2: Coarse features from DGDVs
python src/models/run_classification.py --features coarse --classifier all --split train_test

# Option 3: Both feature types combined
python src/models/run_classification.py --features both --classifier all --split train_test

# Option 4: Cross-validation instead of train/test split
python src/models/run_classification.py --features gcm --classifier all --split cv
```

**What happens:**
1. Loads GCM/DGDV features from individual files
2. Loads labels from dataset (extracted from directory structure)
3. Splits data: 60% train, 20% validation, 20% test (stratified)
4. Trains Random Forest classifier
5. Trains SVM classifier
6. Evaluates both on test set
7. Saves models, metrics, confusion matrices

**Expected output:**
```
============================================================
Malware Family Classification Pipeline
============================================================

[1/5] Loading features and labels...
Loading 5000 GCM files from: data/GCMs/individual
  GCM features shape: (5000, 666)
  Loaded 5000 samples with 666 features
  Feature type: gcm
  Number of classes: 5

[2/5] Training RF classifier...
[3/5] Train/Test Split (60/20/20)...
  Train: 3000 samples
  Validation: 1000 samples
  Test: 1000 samples
Fitted RF classifier on 3000 samples with 666 features

[4/5] Evaluating on test set...
RF - GCM Results:
  Accuracy:        0.8870
  Macro F1:        0.8879
  Macro AUC:       0.9815
...

[5/5] Generating comparison report...
Classification pipeline completed successfully!
```

**Time estimate:** 5-15 minutes for classification

**Output files:**
- Models: `data/results/models/rf_gcm_model.pkl`, `svm_gcm_model.pkl`
- Metrics: `data/results/classification_results/*.json`, `*.csv`
- Confusion matrices: `data/results/classification_results/confusion_matrices/*.csv`, `*.png`
- Comparison: `data/results/classification_results/comparison_report.txt`

#### Step 6: View Results

```bash
# View comparison report
cat data/results/classification_results/comparison_report.txt

# Or open JSON results
cat data/results/classification_results/rf_gcm_results.json
```

### Quick Test (Small Subset)

To test the pipeline on a smaller subset first:

1. **Modify `run_pipeline.py`** to process only first N graphs (e.g., 100)
2. **Run pipeline:**
   ```bash
   python run_pipeline.py
   ```
3. **Test classification:**
   ```bash
   python src/models/run_classification.py --features gcm --classifier rf
   ```

### Common Issues and Solutions

**Issue: "UCL counter not found"**
- Solution: Compile the counter first (Step 2)

**Issue: "Dataset directory not found"**
- Solution: Ensure `data/malnet_tiny/` exists with graph files

**Issue: "Out of memory"**
- Solution: The pipeline uses incremental saving, but ensure you have enough RAM (16GB+ recommended)

**Issue: "No module named 'src.models'"**
- Solution: Run from project root directory, not from `src/` directory

**Issue: "GCM directory not found"**
- Solution: Run feature computation first (Step 4), or specify correct path with `--gcm_dir`

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

### Run Individual Steps (Advanced)

If you need to run steps separately or use combined files:

**1. Compute DGDVs Only:**
```bash
python run_pipeline.py
# This computes DGDVs, applies orbit reduction, and computes GCMs automatically
# Individual files are saved to data/DGDVs/individual/ and data/GCMs/individual/
```

**2. Apply Orbit Reduction (if using combined files):**
```bash
python src/orbit_reduction.py \
    --input data/DGDVs/all_dgdvs_3_4node.pkl \
    --output data/DGDVs/all_dgdvs_3_4node_reduced.pkl
```

**3. Compute GCMs from Combined DGDV File:**
```bash
python src/compute_gcms.py \
    --input data/DGDVs/all_dgdvs_3_4node_reduced.pkl \
    --output data/GCMs/all_gcms_3_4node_reduced.pkl \
    --save-individual
# The --save-individual flag saves individual GCM files automatically
```

**4. Run Classification with Individual Files (Recommended):**
```bash
# Uses individual files by default (GitHub-friendly, no large combined files)
python src/models/run_classification.py --features gcm --classifier all
```

**5. Run Classification with Combined Files (if available):**
```bash
# Specify combined file paths explicitly
python src/models/run_classification.py \
    --features gcm \
    --gcm_file data/GCMs/all_gcms_3_4node_reduced.pkl \
    --classifier all
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
- `data/DGDVs/individual/graph_XXXXX.pkl` - Individual DGDV files (one per graph)
- `data/DGDVs/metadata/processing_metadata_3_4node.json` - Processing metadata
- `data/DGDVs/metadata/orbit_reduction_3_4node.json` - Orbit reduction statistics

**GCM Files:**
- `data/GCMs/individual/graph_XXXXX.pkl` - Individual GCM files (one per graph)

**Classification Results:**
- `data/results/models/` - Trained classifier models
- `data/results/classification_results/` - Evaluation metrics and reports
- `data/results/classification_results/confusion_matrices/` - Confusion matrices

### Loading Individual Files

```python
import pickle

# Load individual DGDV
with open('data/DGDVs/individual/graph_00042.pkl', 'rb') as f:
    dgdv = pickle.load(f)

# Load individual GCM
with open('data/GCMs/individual/graph_00042.pkl', 'rb') as f:
    gcm = pickle.load(f)
```

### Using Individual Files for Classification

The classification pipeline automatically uses individual files by default (GitHub-friendly):

```python
from src.models.feature_extraction import load_features_from_files

# Automatically loads from individual files
features, labels, label_mapping, feature_type = load_features_from_files(
    gcm_dir='data/GCMs/individual',  # Uses individual GCM files
    dgdv_dir='data/DGDVs/individual',  # Uses individual DGDV files
    data_dir='data/malnet_tiny'
)
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

## Classification

The pipeline includes a complete classification system for malware family classification using Random Forest and Support Vector Machine classifiers.

### Feature Types

1. **GCM Features**: Graphlet Correlation Matrix vectors (graph-level features)
   - Captures correlations between orbits across nodes
   - Typically 200-700 features after orbit reduction

2. **Coarse Features**: Aggregate statistics from DGDVs (graph-level features)
   - Mean, std, sum, min, max per orbit across nodes
   - 5 statistics × n_orbits features

3. **Both**: Concatenated GCM and coarse features

### Classifiers

- **Random Forest (RF)**: Ensemble of decision trees
- **Support Vector Machine (SVM)**: RBF kernel with probability estimates

### Evaluation

The classification pipeline provides comprehensive evaluation:

- **Metrics**: Accuracy, Macro-F1, Weighted-F1, Macro-AUC, per-class metrics
- **Confusion Matrices**: CSV and visualization plots
- **Results**: Saved as JSON and CSV files
- **Comparison Reports**: Side-by-side comparison of different configurations

### Usage Examples

```bash
# Basic classification with GCM features
python src/models/run_classification.py --features gcm --classifier rf

# Specify individual file directories explicitly
python src/models/run_classification.py \
    --features gcm \
    --gcm_dir data/GCMs/individual \
    --classifier all

# Use coarse features from DGDVs
python src/models/run_classification.py \
    --features coarse \
    --dgdv_dir data/DGDVs/individual \
    --classifier rf

# Cross-validation instead of train/test split
python src/models/run_classification.py \
    --features gcm \
    --classifier all \
    --split cv
```

### Classification Results

After running classification, results are saved to `data/results/`:

- **Models**: `data/results/models/` - Trained classifier models (`.pkl`)
- **Metrics**: `data/results/classification_results/` - JSON and CSV files
- **Confusion Matrices**: `data/results/classification_results/confusion_matrices/` - CSV and PNG files
- **Comparison Report**: `data/results/classification_results/comparison_report.txt`

### Programmatic Usage

```python
from src.models.classification import ClassificationPipeline
from src.models.feature_extraction import load_features_from_files
from src.models.evaluation import compute_metrics, print_evaluation_summary

# Load features from individual files
features, labels, label_mapping, feature_type = load_features_from_files(
    gcm_dir='data/GCMs/individual',
    data_dir='data/malnet_tiny'
)

# Initialize and train classifier
pipeline = ClassificationPipeline(random_state=42)
pipeline.fit(features[:3000], labels[:3000], classifier_name='rf')

# Evaluate
y_pred = pipeline.predict(features[3000:], classifier_name='rf')
y_proba = pipeline.predict_proba(features[3000:], classifier_name='rf')

metrics = compute_metrics(
    labels[3000:], y_pred, y_proba,
    labels=[label_mapping[i] for i in sorted(label_mapping.keys())]
)

print_evaluation_summary(metrics)
```

## Memory Management

The pipeline uses incremental saving to handle large datasets:

- **Individual files**: Each DGDV/GCM saved immediately after computation
- **Checkpointing**: Progress saved periodically to prevent data loss
- **Resume capability**: Can resume from last checkpoint if interrupted
- **Batch processing**: Large operations split into batches
- **GitHub-friendly**: Individual files avoid large combined files that exceed GitHub limits

## Quick Reference

### Essential Commands

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Compile UCL counter
g++ -O3 -o Directed_Graphlet_Counter_v3 Directed_Graphlet_Counter_v3.cpp

# 3. Compute features (DGDVs + GCMs)
python run_pipeline.py

# 4. Run classification
python src/models/run_classification.py --features gcm --classifier all
```

### File Locations

- **Input**: `data/malnet_tiny/malnet-graphs-tiny/` - Graph edgelist files
- **DGDVs**: `data/DGDVs/individual/graph_XXXXX.pkl` - Individual DGDV files
- **GCMs**: `data/GCMs/individual/graph_XXXXX.pkl` - Individual GCM files
- **Models**: `data/results/models/` - Trained classifier models
- **Results**: `data/results/classification_results/` - Evaluation metrics and reports

### Expected Results

After running the complete pipeline, you should see:

- **5,000 DGDV files** in `data/DGDVs/individual/`
- **5,000 GCM files** in `data/GCMs/individual/`
- **Trained models** in `data/results/models/`
- **Classification accuracy**: ~88-90% with GCM features
- **Macro F1**: ~88-90%
- **Macro AUC**: ~98%

## References

- Research Paper: "Classifying Malware Families Using Graphlet-Based Topological Signatures from MalNet-Tiny"
- UCL Directed Graphlet Counter: http://www0.cs.ucl.ac.uk/staff/natasa/DGCD/index.html
- Dataset: https://github.com/nd7141/graph_datasets
