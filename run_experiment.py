"""
Production-Ready Federated Learning Framework
Comprehensive FL orchestration with enterprise-grade features
Updated: Includes test evaluation and SystemAware weighted aggregation
"""

import gc
import logging
import time
import random
import os
import json
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import copy

from fl import coordinator
from global_args import benchmark_preprocess, read_args, override_args, single_preprocess
from global_utils import avg_value, print_filtered_args, setup_logger, setup_seed
from datapreprocessor.data_utils import load_data, split_dataset, get_transform
from fl.server import Server
from plot_utils import plot_accuracy
from fl.models.model_utils import initialize_model_properly, model2vec
from fl.models.resnet import ResNet18
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from evaluation import evaluate_model as evaluate_detailed
from fl.client import Client

# ------------------------------
# Entry Point
# ------------------------------

if __name__ == "__main__":
    args, cli_args = read_args()
    override_args(args, cli_args)
    args = single_preprocess(args)

    # Set random seed for reproducibility
    setup_seed(args.seed)

    # Ensure attack is set to NoAttack if num_adv is 0 to prevent AssertionError
    if args.num_adv == 0:
        args.attack = "NoAttack"

    # Set device
    args.device = 'cuda' if torch.cuda.is_available() and getattr(args, 'gpu_idx', None) else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu_idx[0])
    print(f"Using device: {args.device}")

    # Set num_channels and num_dims based on dataset for model initialization
    if args.dataset.lower() == "cifar10":
        args.num_channels = 3
        args.num_dims = 32
    elif args.dataset.lower() == "mnist":
        args.num_channels = 1
        args.num_dims = 28

    # Prepare data with correct transforms
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

    # Create clients
    client_indices = split_dataset(train_dataset, args.num_clients, args.distribution, args.num_shards)
    clients = coordinator.init_clients(args, client_indices, train_dataset, test_dataset)

    # Initialize server
    server = Server(args, clients, test_dataset, train_dataset)
    coordinator.set_fl_algorithm(args, server, clients)

    # Main FL loop
    print("\n" + "="*60)
    print("FEDERATED LEARNING - COORDINATOR MODE")
    print("="*60)
    print(f"Algorithm: {args.algorithm}")
    print(f"Aggregator: {server.aggregation_method}")
    print(f"Clients: {len(clients)}")
    print(f"Rounds: {args.epochs}")
    print(f"Local Epochs: {args.local_epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Device: {args.device}")
    print("="*60)

    best_accuracy = 0
    best_model_state = None

    for round_num in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}/{args.epochs}")
        print(f"{'='*60}")

        # Select clients
        selected_clients_indices = random.sample(range(len(clients)), min(args.clients_per_round, len(clients)))
        selected_clients = [clients[i] for i in selected_clients_indices]

        # Distribute global model to selected clients
        for client in selected_clients:
            client.load_global_model(model2vec(server.get_global_model()))
        # Clients perform local training and fetch updates
        for client in selected_clients:
            client.local_training()
            client.fetch_updates() # This will store the client's model in client.model

        # Server collects updates
        server.collect_updates(round_num, active_clients=selected_clients)

        # Server aggregates updates
        server.aggregation()

        # Server updates global model
        server.update_global()

        # Evaluate global model
        metrics = coordinator.evaluate(server, test_dataset, args, round_num)
        current_accuracy = metrics["test_acc"]

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_model_state = copy.deepcopy(server.get_global_model().state_dict())
            # Save best model
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_model_state, output_dir / 'best_global_model.pth')
            print(f"\n🎯 NEW BEST ACCURACY: {best_accuracy:.4f}")

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"BEST ACCURACY: {best_accuracy:.4f}")
    print(f"{'='*60}\n")

    # Load best model if available
    if best_model_state is not None:
        final_model = server.get_global_model()
        final_model.load_state_dict(best_model_state)
    else:
        final_model = server.get_global_model() # Return the last model if no improvement

    print(f"\n🎯 FINAL BEST ACCURACY: {best_accuracy:.4f}")
