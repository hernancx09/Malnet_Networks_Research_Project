#!/usr/bin/env python3
"""
Split combined GCM file into individual GCM files
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from compute_dgdvs_malnet import split_gcm_file_into_individual

def main():
    # Default paths
    combined_file = "data/gdvs/gcms/all_gcms_3_4node.pkl"
    individual_dir = "data/gdvs/gcms/individual"
    
    # Allow command line override
    if len(sys.argv) > 1:
        combined_file = sys.argv[1]
    if len(sys.argv) > 2:
        individual_dir = sys.argv[2]
    
    print(f"Splitting GCM file: {combined_file}")
    print(f"Output directory: {individual_dir}")
    print()
    
    split_gcm_file_into_individual(
        combined_file=combined_file,
        individual_dir=individual_dir,
        delete_existing=True
    )

if __name__ == "__main__":
    main()

