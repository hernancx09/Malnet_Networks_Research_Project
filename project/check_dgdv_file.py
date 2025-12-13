#!/usr/bin/env python3
"""Check if combined DGDV file exists and is complete"""
import pickle
from pathlib import Path

dgdv_file = Path("data/gdvs/gdvs/all_dgdvs_3_4node.pkl")

if dgdv_file.exists():
    try:
        print(f"Checking file: {dgdv_file}")
        print(f"File size: {dgdv_file.stat().st_size / (1024**2):.1f} MB")
        with open(dgdv_file, 'rb') as f:
            dgdvs = pickle.load(f)
        print(f"✅ File exists with {len(dgdvs)} DGDVs")
        if len(dgdvs) == 5000:
            print("✅ File is complete (5000/5000)")
        else:
            print(f"⚠️  File is incomplete ({len(dgdvs)}/5000)")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
else:
    print(f"❌ File does not exist: {dgdv_file}")

