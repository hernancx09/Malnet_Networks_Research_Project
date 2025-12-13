# Installing CUDA Toolkit for WSL

CuPy is installed in the virtual environment, but you need to install the CUDA toolkit in WSL to provide the runtime libraries.

## Quick Install (Recommended)

Run this command in WSL (you'll need to enter your password):

```bash
sudo apt update
sudo apt install -y cuda-toolkit-12-9
```

After installation, you may need to add CUDA to your PATH. Add these lines to your `~/.bashrc`:

```bash
export PATH=/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
```

Then reload:
```bash
source ~/.bashrc
```

## Alternative: Use Windows CUDA (if available)

If you have CUDA installed on Windows, WSL can sometimes access it. Check if CUDA libraries are available:

```bash
ls /usr/lib/wsl/lib/ | grep -i cuda
```

## Verify Installation

After installing CUDA toolkit, test CuPy:

```bash
cd /mnt/c/Users/baraj/Desktop/Classes/Networks/Malnet_Networks_Research_Project/project
source venv/bin/activate
python3 test_cupy.py
```

You should see "✅ CuPy is working correctly!" if everything is set up properly.

