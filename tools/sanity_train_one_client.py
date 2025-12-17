import sys
import os

# Add the project root to the Python path to find 'fl' module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root) # Insert at the beginning to ensure it's found first

import torch
import torch.nn as nn
import torch.optim as optim

from fl.client import Client
from fl.models import get_model
from datapreprocessor.data_utils import load_data, split_dataset, get_transform
from global_args import read_args, override_args, single_preprocess
from fl.models.model_utils import initialize_model_properly, model2vec # Import for model initialization and vector conversion
from torch.utils.data import DataLoader # Import for DataLoader
from torchvision import datasets, transforms # Import for datasets and transforms
# The global_args.py handles argument parsing, so we don't need a separate parser here.
# The script will receive arguments via sys.argv, which read_args() processes.
args, cli_args = read_args()
override_args(args, cli_args)

args = single_preprocess(args)


# Set num_channels and num_dims based on dataset for model initialization
if args.dataset.lower() == "cifar10":
    args.num_channels = 3
    args.num_dims = 32
elif args.dataset.lower() == "mnist":
    args.num_channels = 1
    args.num_dims = 28

train_transform, test_transform = get_transform(args)

# Load dataset
if args.dataset.lower() == "cifar10":
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
elif args.dataset.lower() == "mnist":
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
else:
    raise ValueError(f"Dataset {args.dataset} not supported for direct execution.")


client_indices = split_dataset(train_dataset, 1, args.distribution, args.num_shards)

# Create a Subset for the client's training data
client_train_dataset = torch.utils.data.Subset(train_dataset, client_indices[0])

# instantiate a client
c = Client(args=args, worker_id=0, train_dataset=client_train_dataset, test_dataset=test_dataset)

# ensure criterion, optimizer, loader exist
print("criterion:", c.criterion_fn)
# The client's local_training method now handles its own optimizer and scheduler
# So, we don't need to explicitly set c.optimizer or c.lr_scheduler here.
# The `hetero_lr` is also handled internally by the new local_training.

# Initialize the client's model properly
c.model = initialize_model_properly(c.model)

# Load global model (dummy for single client test)
# The get_model(args) here will create a new model, which is then vectorized.
# This is fine for a sanity check, as we just need a starting point.
dummy_global_model = get_model(args)
dummy_global_model = initialize_model_properly(dummy_global_model) # Initialize dummy model too

# Debug prints to inspect model parameters and vector
print(f"DEBUG: dummy_global_model state_dict keys: {dummy_global_model.state_dict().keys()}")
total_params = sum(p.numel() for p in dummy_global_model.parameters())
print(f"DEBUG: Total parameters in dummy_global_model: {total_params}")

vectorized_params = model2vec(dummy_global_model)
print(f"DEBUG: Size of vectorized_params: {vectorized_params.size}")

c.load_global_model(vectorized_params)


if __name__ == '__main__':
    print("[Client 0] Starting local training...")
    acc, loss = c.local_training(local_epochs=5) # run local training
    print("local training acc, loss:", acc, loss)

    # Perform client-level test after local training
    test_acc, test_loss = c.client_test(c.model, c.test_dataset)
    print(f"[Client 0] Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")
    print("[Client 0] Training and evaluation complete ✅")
