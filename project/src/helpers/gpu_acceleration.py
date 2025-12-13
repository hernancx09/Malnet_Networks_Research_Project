#!/usr/bin/env python3
"""
GPU Acceleration utilities for DGDV computation using CuPy
Provides GPU-accelerated matrix operations as a drop-in replacement for NumPy
"""

import numpy as np
from typing import Optional

# Try to import CuPy, fall back to NumPy if not available
try:
    import cupy as cp
    GPU_AVAILABLE = True
    try:
        # Test GPU availability - try a simple operation
        _ = cp.array([1, 2, 3])
        # Try a more complex operation to ensure CUDA runtime is available
        _ = cp.zeros((10, 10))
        GPU_ENABLED = True
    except (RuntimeError, FileNotFoundError, OSError) as e:
        # CUDA runtime not available (DLL not found, etc.)
        GPU_ENABLED = False
        GPU_ERROR = str(e)
    except Exception as e:
        GPU_ENABLED = False
        GPU_ERROR = str(e)
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    GPU_ENABLED = False
    GPU_ERROR = None


def get_array_module(use_gpu: bool = True):
    """
    Get the appropriate array module (CuPy or NumPy)
    
    Args:
        use_gpu: Whether to use GPU if available
    
    Returns:
        Array module (cupy or numpy)
    """
    if use_gpu and GPU_ENABLED:
        try:
            # Test if we can actually use CuPy
            _ = cp.array([1])
            return cp
        except (RuntimeError, FileNotFoundError, OSError):
            # CUDA runtime error - fall back to NumPy
            return np
    return np


def to_gpu(array: np.ndarray, use_gpu: bool = True) -> np.ndarray:
    """
    Convert NumPy array to GPU (CuPy) array if GPU is enabled
    
    Args:
        array: NumPy array
        use_gpu: Whether to use GPU
    
    Returns:
        GPU array (CuPy) or original NumPy array
    """
    if use_gpu and GPU_ENABLED and cp is not None:
        try:
            return cp.asarray(array)
        except (RuntimeError, FileNotFoundError, OSError):
            # CUDA runtime error - return CPU array
            return array
    return array


def to_cpu(array) -> np.ndarray:
    """
    Convert GPU (CuPy) array back to NumPy array
    
    Args:
        array: GPU or CPU array
    
    Returns:
        NumPy array
    """
    if GPU_ENABLED and cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def zeros(shape, dtype=np.int64, use_gpu: bool = True):
    """
    Create zero array on GPU or CPU
    
    Args:
        shape: Array shape
        dtype: Data type
        use_gpu: Whether to use GPU
    
    Returns:
        Zero array (always returns NumPy array for compatibility)
    """
    xp = get_array_module(use_gpu)
    try:
        result = xp.zeros(shape, dtype=dtype)
        # Always convert back to NumPy for compatibility
        if xp is cp:
            return cp.asnumpy(result)
        return result
    except (RuntimeError, FileNotFoundError, OSError):
        # Fall back to NumPy on CUDA errors
        return np.zeros(shape, dtype=dtype)


def corrcoef(matrix, use_gpu: bool = True):
    """
    Compute correlation coefficient matrix on GPU or CPU
    
    Args:
        matrix: Input matrix (nodes × features)
        use_gpu: Whether to use GPU
    
    Returns:
        Correlation matrix (always NumPy for compatibility)
    """
    try:
        xp = get_array_module(use_gpu)
        
        # Convert to GPU if needed
        if use_gpu and GPU_ENABLED and cp is not None and xp is cp:
            try:
                if isinstance(matrix, np.ndarray):
                    matrix_gpu = cp.asarray(matrix)
                else:
                    matrix_gpu = matrix
            except (RuntimeError, FileNotFoundError, OSError):
                # CUDA error - fall back to CPU
                xp = np
                matrix_gpu = matrix
        else:
            matrix_gpu = matrix
        
        # Compute correlation
        # Remove zero-variance columns
        var = xp.var(matrix_gpu, axis=0)
        non_zero = var > 0
        
        if xp.sum(non_zero) == 0:
            return np.array([])
        
        matrix_filtered = matrix_gpu[:, non_zero]
        
        # Compute correlation
        corr = xp.corrcoef(matrix_filtered.T)
        
        # Convert back to CPU if needed
        if xp is cp:
            try:
                return cp.asnumpy(corr)
            except (RuntimeError, FileNotFoundError, OSError):
                # If conversion fails, compute on CPU
                return np.corrcoef(matrix[:, np.var(matrix, axis=0) > 0].T)
        
        return corr
    except (RuntimeError, FileNotFoundError, OSError):
        # Fall back to CPU on any CUDA errors
        non_zero_orbits = np.var(matrix, axis=0) > 0
        if non_zero_orbits.sum() == 0:
            return np.array([])
        return np.corrcoef(matrix[:, non_zero_orbits].T)


def hstack(arrays, use_gpu: bool = True):
    """
    Stack arrays horizontally on GPU or CPU
    
    Args:
        arrays: List of arrays to stack
        use_gpu: Whether to use GPU
    
    Returns:
        Horizontally stacked array (always NumPy for compatibility)
    """
    try:
        xp = get_array_module(use_gpu)
        
        # Convert all arrays to same device
        if use_gpu and GPU_ENABLED and cp is not None and xp is cp:
            try:
                arrays_gpu = [cp.asarray(arr) if isinstance(arr, np.ndarray) else arr 
                             for arr in arrays]
                result = xp.hstack(arrays_gpu)
                return cp.asnumpy(result)  # Convert back to CPU for compatibility
            except (RuntimeError, FileNotFoundError, OSError):
                # CUDA error - fall back to CPU
                return np.hstack(arrays)
        
        return np.hstack(arrays)
    except (RuntimeError, FileNotFoundError, OSError):
        # Fall back to NumPy on any CUDA errors
        return np.hstack(arrays)


def get_gpu_info() -> dict:
    """
    Get GPU information if available
    
    Returns:
        Dictionary with GPU information
    """
    if not GPU_AVAILABLE:
        return {'available': False, 'enabled': False}
    
    if not GPU_ENABLED:
        error_msg = GPU_ERROR if 'GPU_ERROR' in globals() and GPU_ERROR else 'GPU not accessible'
        return {'available': True, 'enabled': False, 'error': error_msg}
    
    try:
        mempool = cp.get_default_memory_pool()
        meminfo = cp.cuda.runtime.memGetInfo()
        
        return {
            'available': True,
            'enabled': True,
            'device': cp.cuda.Device().id,
            'device_name': cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name'].decode(),
            'free_memory_mb': meminfo[0] / 1024**2,
            'total_memory_mb': meminfo[1] / 1024**2,
            'used_memory_mb': mempool.used_bytes() / 1024**2
        }
    except (RuntimeError, FileNotFoundError, OSError) as e:
        return {
            'available': True,
            'enabled': False,
            'error': f'CUDA runtime error: {str(e)}'
        }
    except Exception as e:
        return {
            'available': True,
            'enabled': False,
            'error': str(e)
        }


def print_gpu_status():
    """Print GPU status information"""
    info = get_gpu_info()
    
    print("="*60)
    print("GPU Acceleration Status")
    print("="*60)
    
    if not info['available']:
        print("❌ CuPy not installed")
        print("   Install with: pip install cupy-cuda11x (or cupy-cuda12x)")
        print("   Or use CPU-only mode (default)")
    elif not info['enabled']:
        print("⚠️  GPU not accessible - falling back to CPU mode")
        if 'error' in info:
            error_msg = info['error']
            if 'nvrtc' in error_msg.lower() or 'dll' in error_msg.lower():
                print(f"   Error: CUDA runtime DLL not found")
                print("   Solution: Install CUDA Toolkit or set CUDA_PATH environment variable")
                print("   See GPU_SETUP.md for troubleshooting")
            else:
                print(f"   Error: {error_msg}")
    else:
        print("✅ GPU acceleration enabled")
        print(f"   Device: {info['device_name']}")
        print(f"   Memory: {info['free_memory_mb']:.0f} MB free / {info['total_memory_mb']:.0f} MB total")
        print(f"   Used: {info['used_memory_mb']:.0f} MB")
    
    print("="*60)

