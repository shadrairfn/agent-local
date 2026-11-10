"""
=============================================================================
ENVIRONMENT & HARDWARE AUDIT VERIFIER (GEMASTIK XIX & KAGGLE STANDARDS)
=============================================================================
Audits Python, PyTorch, CUDA, GPU VRAM (RTX 3050), and essential ML/NLP packages.
"""

import sys
import os

print("=" * 70)
print("[HARDWARE & ENVIRONMENT VERIFICATION AUDIT]")
print("=" * 70)

# 1. Python Version
print(f" * Python Version:    {sys.version.split()[0]}")

# 2. PyTorch & CUDA Audit
try:
    import torch
    print(f" * PyTorch Version:   {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f" * CUDA Available:    {cuda_avail}")
    if cuda_avail:
        device_name = torch.cuda.get_device_name(0)
        vram_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2))
        print(f" * GPU Device:        {device_name}")
        print(f" * Total VRAM:        {vram_mb} MB")
        if "3050" in device_name:
            print("   -> [HARDWARE MATCH] NVIDIA GeForce RTX 3050 detected.")
            print("   -> Recommended BATCH_SIZE=8 (train), MAX_TOKEN_LEN=384, FP16=True.")
    else:
        print("   -> [FALLBACK] CPU only mode. Use BATCH_SIZE=4 and small models.")
except ImportError:
    print("   -> [ERROR] PyTorch is not installed!")

# 3. Essential Scientific & ML Packages
packages = [
    ("pandas", "Data manipulation"),
    ("numpy", "Numerical computing"),
    ("scipy", "Scientific optimization"),
    ("sklearn", "Scikit-learn algorithms & metrics"),
    ("lightgbm", "Gradient boosted decision trees"),
    ("transformers", "Hugging Face pretrained models"),
    ("tokenizers", "Fast subword tokenizers"),
    ("accelerate", "PyTorch training acceleration")
]

print("\n" + "-" * 70)
print(" * Checking Essential Libraries:")
all_ok = True
for pkg, desc in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "installed")
        print(f"   [PASS] {pkg:<15} ({ver:<10}) - {desc}")
    except ImportError:
        print(f"   [FAIL] {pkg:<15} (NOT FOUND)   - {desc}")
        all_ok = False

print("-" * 70)
if all_ok:
    print("[ALL CHECKS PASSED] Environment is 100% ready for GEMASTIK XIX competition!")
else:
    print("[WARNING] Some packages are missing. Run: pip install -r requirements.txt")
print("=" * 70)

