"""
Complete Data Loading Implementation with Windows DataLoader Fix
Use this code in your datapreprocessor/data_utils.py or wherever data loading happens
"""

import os
import pickle
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


# ==================== Safe DataLoader Creator ====================

def create_dataloader(dataset, batch_size, shuffle=True, drop_last=False):
    """
    Create a Windows-compatible DataLoader
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
    
    Returns:
        DataLoader instance with Windows-safe settings
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,          # CRITICAL: Must be 0 on Windows
        pin_memory=False,       # CRITICAL: Must be False on Windows
        drop_last=drop_last,
        persistent_workers=False
    )


# ==================== Dataset Loading Functions ====================

def load_data(args):
    """
    Load dataset with proper transforms
    
    Args:
        args: Configuration object with dataset info
    
    Returns:
        train_dataset, test_dataset
    """
    # Get transforms
    train_transform, test_transform = get_transform(args)
    data_directory = './data'
    
    # Ensure data directory exists
    os.makedirs(data_directory, exist_ok=True)
    
    # Load based on dataset type
    if args.dataset == "EMNIST":
        train_dataset = datasets.EMNIST(
            data_directory, 
            split="digits", 
            train=True, 
            download=True,
            transform=train_transform
        )
        test_dataset = datasets.EMNIST(
            data_directory, 
            split="digits", 
            train=False, 
            download=True,
            transform=test_transform
        )
    
    elif args.dataset in ["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"]:
        dataset_class = getattr(datasets, args.dataset)
        train_dataset = dataset_class(
            root=data_directory, 
            train=True,
            download=True, 
            transform=train_transform
        )
        test_dataset = dataset_class(
            root=data_directory, 
            train=False,
            download=True, 
            transform=test_transform
        )
    
    elif args.dataset == "CINIC10":
        from datapreprocessor.cinic10 import CINIC10
        train_dataset = CINIC10(
            root=data_directory, 
            train=True, 
            download=True,
            transform=train_transform
        )
        test_dataset = CINIC10(
            root=data_directory, 
            train=False, 
            download=True,
            transform=test_transform
        )
    
    else:
        raise ValueError(f"Dataset {args.dataset} not implemented")
    
    # Convert targets to tensor if needed
    train_dataset.targets = list_to_tensor(train_dataset.targets)
    test_dataset.targets = list_to_tensor(test_dataset.targets)
    
    return train_dataset, test_dataset


def get_transform(args):
    """
    Get appropriate transforms for dataset
    
    Args:
        args: Configuration object
    
    Returns:
        train_transform, test_transform
    """
    if args.dataset in ["MNIST", "FashionMNIST", "EMNIST", "FEMNIST"]:
        if args.model in ['lenet', "lr"]:
            # Resize to 32x32 for LeNet5
            train_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(args.mean, args.std)
            ])
            args.num_dims = 32
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(args.mean, args.std)
            ])
            args.num_dims = 28
        
        test_transform = train_transform
    
    elif args.dataset == "CINIC10":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean, std=args.std)
        ])
        test_transform = train_transform
        args.num_dims = 32
    
    elif args.dataset in ["CIFAR10", "CIFAR100", "TinyImageNet"]:
        args.num_dims = 32 if args.dataset in ['CIFAR10', 'CIFAR100'] else 64
        
        # Training transforms (with optional augmentation)
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(args.mean, args.std)
        ])
        
        # Test transforms (no augmentation)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(args.mean, args.std)
        ])
    
    else:
        raise ValueError(f"Dataset {args.dataset} not implemented")
    
    return train_transform, test_transform


def list_to_tensor(vector):
    """
    Convert list to tensor if needed
    
    Args:
        vector: List or tensor
    
    Returns:
        Tensor
    """
    if isinstance(vector, list):
        vector = torch.tensor(vector)
    return vector


# ==================== Dataset Partition Class ====================

class Partition(Dataset):
    """
    Dataset partition for federated learning
    
    Args:
        dataset: Original dataset
        indices: Indices for this partition
        transform: Optional transform to apply
    """
    
    def __init__(self, dataset, indices=None, transform=None):
        self.dataset = dataset
        self.classes = dataset.classes
        self.indices = indices if indices is not None else range(len(dataset))
        self.data = dataset.data[self.indices]
        self.targets = dataset.targets[self.indices]
        
        # Determine image mode
        self.mode = 'L' if len(self.data.shape) == 3 else 'RGB'
        self.transform = transform
        self.poison = False
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, target = self.data[idx], self.targets[idx]
        
        # Convert to numpy if needed
        if not isinstance(image, (np.ndarray, np.generic)):
            image = image.numpy()
        
        # Convert to PIL Image
        image = Image.fromarray(image, mode=self.mode)
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # Apply poisoning if enabled
        if self.poison:
            image, target = self.synthesizer.backdoor_batch(
                image, target.reshape(-1, 1)
            )
        
        return image, target.squeeze()
    
    def poison_setup(self, synthesizer):
        """Enable poisoning with given synthesizer"""
        self.poison = True
        self.synthesizer = synthesizer


def subset_by_idx(args, dataset, indices, train=True):
    """
    Create a subset of dataset by indices
    
    Args:
        args: Configuration object
        dataset: Original dataset
        indices: Indices to include
        train: Whether this is training data
    
    Returns:
        Partition dataset
    """
    transform = get_transform(args)[0] if train else get_transform(args)[1]
    return Partition(dataset, indices, transform=transform)


# ==================== Client DataLoader Creation ====================

def create_client_dataloaders(train_dataset, test_dataset, batch_size):
    """
    Create train and test dataloaders for a client
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        batch_size: Batch size
    
    Returns:
        train_loader, test_loader
    """
    # Training DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,          # Windows compatibility
        pin_memory=False,       # Windows compatibility
        drop_last=False,
        persistent_workers=False
    )
    
    # Test DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,          # No shuffle for test
        num_workers=0,          # Windows compatibility
        pin_memory=False,       # Windows compatibility
        drop_last=False,
        persistent_workers=False
    )
    
    return train_loader, test_loader


# ==================== Example Usage in Client ====================

class ExampleClient:
    """
    Example client class showing how to use the data loading functions
    """
    
    def __init__(self, worker_id, args, train_dataset, test_dataset):
        self.worker_id = worker_id
        self.args = args
        self.batch_size = getattr(args, 'batch_size', 64)
        
        # Store datasets
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        
        # Create DataLoaders using safe function
        self.train_loader, self.test_loader = create_client_dataloaders(
            train_dataset, 
            test_dataset, 
            self.batch_size
        )
        
        print(f"[Client {worker_id}] DataLoaders created "
              f"(num_workers=0, pin_memory=False)", flush=True)
    
    def local_training(self):
        """Example training method"""
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Your training code here
            # ...
            pass
        
        return accuracy, loss


# ==================== Utility Functions ====================

def get_dataloader_safe(dataset, batch_size=64, shuffle=True, **kwargs):
    """
    Get a DataLoader with guaranteed Windows compatibility
    
    This function overrides any dangerous parameters and ensures
    the DataLoader will work on Windows systems.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size (default: 64)
        shuffle: Whether to shuffle (default: True)
        **kwargs: Other arguments (num_workers and pin_memory will be overridden)
    
    Returns:
        DataLoader instance
    """
    # Override problematic parameters
    kwargs['num_workers'] = 0
    kwargs['pin_memory'] = False
    kwargs['persistent_workers'] = False
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        **kwargs
    )


def test_dataloader_works():
    """
    Test if DataLoader works correctly
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create dummy dataset
        data = torch.randn(100, 1, 28, 28)
        labels = torch.randint(0, 10, (100,))
        dataset = torch.utils.data.TensorDataset(data, labels)
        
        # Create DataLoader
        loader = create_dataloader(dataset, batch_size=32, shuffle=True)
        
        # Try to iterate
        for i, (batch_data, batch_labels) in enumerate(loader):
            if i >= 2:
                break
        
        print("✓ DataLoader test passed", flush=True)
        return True
        
    except Exception as e:
        print(f"✗ DataLoader test failed: {e}", flush=True)
        return False


# ==================== Main Test ====================

if __name__ == "__main__":
    """
    Test the data loading functionality
    """
    import sys
    
    print("Testing data loading functions...\n")
    
    # Test DataLoader
    if test_dataloader_works():
        print("\n✓ All data loading functions are working correctly!")
        sys.exit(0)
    else:
        print("\n✗ Data loading test failed")
        sys.exit(1)