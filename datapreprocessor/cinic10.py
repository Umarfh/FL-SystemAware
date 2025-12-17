"""
Fix for DataLoader hanging on Windows
Apply these changes to your client code
"""

# ==================== Option 1: Quick Fix in Client Code ====================

# Find where DataLoader is created in your client.py (probably in __init__ or local_training)
# It likely looks like this:

# BEFORE (causes hanging):
self.train_loader = DataLoader(
    self.train_dataset,
    batch_size=self.batch_size,
    shuffle=True
)

# AFTER (works on Windows):
self.train_loader = DataLoader(
    self.train_dataset,
    batch_size=self.batch_size,
    shuffle=True,
    num_workers=0,          # CRITICAL: Must be 0 on Windows
    pin_memory=False,       # CRITICAL: Must be False on Windows
    persistent_workers=False  # Add this too
)


# ==================== Option 2: Global Patch (Apply in main.py) ====================

# Add this at the VERY TOP of main.py, before any other imports:

import sys
import os

# Force single-threaded mode for Windows
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

print("[PATCH] Applying Windows DataLoader fixes...", flush=True)

import torch
import torch.utils.data

# Store original DataLoader
_OriginalDataLoader = torch.utils.data.DataLoader

class WindowsSafeDataLoader(_OriginalDataLoader):
    """
    DataLoader wrapper that forces safe settings on Windows
    """
    def __init__(self, *args, **kwargs):
        # Force safe settings
        kwargs['num_workers'] = 0
        kwargs['pin_memory'] = False
        kwargs.pop('persistent_workers', None)
        
        # Call original
        super().__init__(*args, **kwargs)
        
        # Add tracking
        if not hasattr(WindowsSafeDataLoader, '_instance_count'):
            WindowsSafeDataLoader._instance_count = 0
        WindowsSafeDataLoader._instance_count += 1

# Replace DataLoader globally
torch.utils.data.DataLoader = WindowsSafeDataLoader

print("[PATCH] ✓ DataLoader patched for Windows compatibility", flush=True)
print("[PATCH] All DataLoaders will use num_workers=0, pin_memory=False", flush=True)
sys.stdout.flush()

# Now import the rest of your code
from global_args import read_args, override_args, single_preprocess, benchmark_preprocess
# ... rest of imports


# ==================== Option 3: Specific Client Fix ====================

# If you have access to the Client class, modify it like this:

class Client:
    def __init__(self, worker_id, args, train_dataset, test_dataset):
        # ... existing code ...
        
        # Create DataLoader with Windows-safe settings
        dataloader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': 0,        # Windows: MUST be 0
            'pin_memory': False,     # Windows: MUST be False
            'drop_last': False
        }
        
        self.train_loader = DataLoader(
            self.train_dataset,
            **dataloader_kwargs
        )
        
        # Test loader (no shuffle for testing)
        test_kwargs = dataloader_kwargs.copy()
        test_kwargs['shuffle'] = False
        
        self.test_loader = DataLoader(
            self.test_dataset,
            **test_kwargs
        )
        
        print(f"[Client {self.worker_id}] DataLoader configured for Windows", flush=True)


# ==================== Option 4: Debug and Fix ====================

# Add this debug code to find ALL DataLoader creations:

import torch.utils.data
import traceback

_original_dataloader_init = torch.utils.data.DataLoader.__init__

def debug_dataloader_init(self, *args, **kwargs):
    """Debug wrapper to track DataLoader creation"""
    print(f"\n[DEBUG] DataLoader created with:", flush=True)
    print(f"  num_workers: {kwargs.get('num_workers', 'default')}", flush=True)
    print(f"  pin_memory: {kwargs.get('pin_memory', 'default')}", flush=True)
    print(f"  Caller:", flush=True)
    for line in traceback.format_stack()[-4:-1]:
        print(f"    {line.strip()}", flush=True)
    
    # Force safe settings
    kwargs['num_workers'] = 0
    kwargs['pin_memory'] = False
    
    return _original_dataloader_init(self, *args, **kwargs)

torch.utils.data.DataLoader.__init__ = debug_dataloader_init


# ==================== Option 5: Command-Line Environment ====================

# Run your script with these environment variables:

# Windows PowerShell:
# $env:OMP_NUM_THREADS=1
# $env:MKL_NUM_THREADS=1  
# $env:PYTHONUNBUFFERED=1
# python -u main.py -config configs/FedOpt_MNIST_config.yaml ...

# Windows CMD:
# set OMP_NUM_THREADS=1
# set MKL_NUM_THREADS=1
# set PYTHONUNBUFFERED=1
# python -u main.py -config configs/FedOpt_MNIST_config.yaml ...


# ==================== Verification Script ====================

def verify_dataloader_settings():
    """
    Run this to verify DataLoader configuration
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    print("\n" + "="*60)
    print("DataLoader Configuration Test")
    print("="*60)
    
    # Create dummy data
    data = torch.randn(100, 10)
    labels = torch.randint(0, 2, (100,))
    dataset = TensorDataset(data, labels)
    
    # Test with safe settings
    print("\nTesting with num_workers=0, pin_memory=False...")
    try:
        loader = DataLoader(
            dataset,
            batch_size=32,
            num_workers=0,
            pin_memory=False
        )
        
        # Try to iterate
        for i, (batch_data, batch_labels) in enumerate(loader):
            print(f"  Batch {i}: {batch_data.shape}")
            if i >= 2:
                break
        
        print("✓ DataLoader works correctly!")
        return True
        
    except Exception as e:
        print(f"✗ DataLoader failed: {e}")
        return False

if __name__ == "__main__":
    verify_dataloader_settings()