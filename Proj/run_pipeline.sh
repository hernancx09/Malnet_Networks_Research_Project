#!/bin/bash

#$ -M hbarajas@nd.edu
#$ -m abe
#$ -pe smp 1
#$ -q long
#$ -N malnet_gdv_gcm_pipeline

# Get AFS token (required for accessing AFS filesystem)
# This uses the Kerberos credential cache passed by UGE
# Note: You may need to run 'aklog' on the login node before submitting
if [ -n "$KRB5CCNAME" ]; then
    echo "Acquiring AFS token from Kerberos credentials..."
    /usr/bin/aklog || echo "Warning: Failed to acquire AFS token, continuing..."
else
    echo "Warning: KRB5CCNAME not set, attempting aklog anyway..."
    /usr/bin/aklog || echo "Warning: aklog failed, continuing anyway..."
fi

# Exit on error (set after AFS token acquisition)
set -e

# Change to project directory
cd /users/hbarajas/Networks/proj/project

# Load required modules (adjust module names based on your cluster's available modules)
# Common options: python/3.9, python3, anaconda3, python/3.10
# Check available modules with: module avail python
module load python/3.12.11  # Using available Python module

# Optional: Activate virtual environment if you have one
# source venv/bin/activate

# Optional: Install/verify dependencies (uncomment if needed on first run)
pip install --user networkx>=3.0 matplotlib>=3.5.0 "numpy>=1.21.0,<2.0"

# Print environment info for debugging
echo "=========================================="
echo "Job Information:"
echo "  Job ID: $JOB_ID"
echo "  Job Name: $JOB_NAME"
echo "  Host: $HOSTNAME"
echo "  Working Directory: $(pwd)"
echo "  Python: $(which python3)"
echo "  Python Version: $(python3 --version)"
echo "=========================================="

# Run the pipeline
python3 run_pipeline.py

echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="

