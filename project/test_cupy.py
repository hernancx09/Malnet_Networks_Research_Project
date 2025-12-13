#!/usr/bin/env python3
"""Test CuPy installation and GPU access"""
import cupy as cp

print("="*60)
print("CuPy Installation Test")
print("="*60)
print(f"CuPy version: {cp.__version__}")
print(f"CUDA version: {cp.cuda.runtime.runtimeGetVersion()}")
print(f"GPU device: {cp.cuda.Device(0).compute_capability}")
meminfo = cp.cuda.Device(0).mem_info
print(f"GPU memory: {meminfo[0] / 1024**3:.2f} GB free / {meminfo[1] / 1024**3:.2f} GB total")

# Test a simple computation
print("\nTesting GPU computation...")
a = cp.array([1, 2, 3, 4, 5])
b = cp.array([5, 4, 3, 2, 1])
c = a + b
print(f"Test computation result: {c.get()}")
print("="*60)
print("✅ CuPy is working correctly!")
print("="*60)

