# Data Directory

## Structure

- `malnet_tiny/` - MalNet-Tiny dataset (5,000 graphs)
- `gdvs/` - Computed Graphlet Degree Vectors
- `gcms/` - Computed Graphlet Correlation Matrices
- `processed/` - Processed/intermediate graph files

## Dataset

The MalNet-Tiny dataset should be in `malnet_tiny/` with the following structure:
```
malnet_tiny/
└── malnet-graphs-tiny/
    ├── addisplay/kuguo/*.edgelist
    ├── adware/airpush/*.edgelist
    ├── benign/benign/*.edgelist
    ├── downloader/jiagu/*.edgelist
    └── trojan/artemis/*.edgelist
```

## Output Files

After running the pipeline:
- `gdvs/all_gdvs.pkl` - All GDV matrices
- `gdvs/all_gcms.pkl` - All GCM vectors
- `gdvs/metadata.json` - Processing information

