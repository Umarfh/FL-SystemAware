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
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import torch.nn.utils # Added for model size calculation

from fl import coordinator
from global_args import read_args, override_args, single_preprocess
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

    # Setup logging
    # Setup logging
    log_dir = Path(args.output) / "run_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger("fl_training", log_dir, level=log_level, stream=True)
    logger.info("Starting Federated Learning Experiment")
    print_filtered_args(args, logger)
    logger.info(f"Output directory specified: {args.output}") # Debug print

    # Set random seed for reproducibility
    setup_seed(args.seed)

    # Ensure attack is set to NoAttack if num_adv is 0 to prevent AssertionError
    if args.num_adv == 0:
        args.attack = "NoAttack"

    # Set device
    args.device = 'cuda' if torch.cuda.is_available() and getattr(args, 'gpu_idx', None) else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu_idx[0])
    logger.info(f"Using device: {args.device}")

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

    # Calculate model size once
    model_size_kb = 0.0
    if server.global_model:
        num_params = sum(p.numel() for p in server.global_model.parameters())
        model_size_kb = (num_params * 4) / 1024.0 # Assuming float32, 4 bytes per parameter
        logger.info(f"[Server] Global model size: {model_size_kb:.2f} KB")

    # Main FL loop
    logger.info("\n" + "="*60)
    logger.info("FEDERATED LEARNING - COORDINATOR MODE")
    logger.info("="*60)
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"Aggregator: {server.aggregation_method}")
    logger.info(f"Clients: {len(clients)}")
    logger.info(f"Rounds: {args.epochs}")
    logger.info(f"Local Epochs: {args.local_epochs}")
    logger.info(f"Learning Rate: {args.learning_rate}")
    logger.info(f"Device: {args.device}")
    logger.info("="*60)

    best_accuracy = 0
    best_model_state = None
    all_round_metrics = [] # List to store metrics for each round

    logger.info("Starting main FL loop...")
    for round_num in range(1, args.epochs + 1):
        round_start_time = time.time() # Start timing for the current round
        logger.info(f"\n{'='*60}")
        logger.info(f"ROUND {round_num}/{args.epochs}")
        logger.info(f"{'='*60}")

        # Store initial rejected count from server metrics
        initial_rejected_count = server.metrics.total_updates_rejected
        logger.debug(f"Initial rejected count: {initial_rejected_count}")

        # Select clients
        selected_clients_indices = random.sample(range(len(clients)), min(args.clients_per_round, len(clients)))
        selected_clients = [clients[i] for i in selected_clients_indices]
        selected_clients_count = len(selected_clients) # N_t
        logger.debug(f"Selected {selected_clients_count} clients for round {round_num}")

        # Distribute global model to selected clients
        for client in selected_clients:
            client.load_global_model(model2vec(server.get_global_model()))
        logger.debug(f"Global model distributed to {selected_clients_count} clients.")

        # Clients perform local training and fetch updates
        for client in selected_clients:
            logger.debug(f"Client {client.client_id}: Starting local training.")
            try:
                client.local_training()
                logger.debug(f"Client {client.client_id}: Local training completed.")
            except Exception as e:
                logger.error(f"Error during client {client.client_id} local training: {e}", exc_info=True)
                # If local training fails, the client won't have an update, so skip to the next client
                continue

            logger.debug(f"Client {client.client_id}: Starting fetch updates.")
            try:
                client.fetch_updates() # This will store the client's model in client.model
                logger.debug(f"Client {client.client_id}: Fetch updates completed.")
            except Exception as e:
                logger.error(f"Error during client {client.client_id} fetch_updates: {e}", exc_info=True)
                # If fetching updates fails, the client won't have a valid update, so skip
                continue
        logger.debug("All selected clients attempted local training and update fetching.")

        # Server collects updates
        collected_count = server.collect_updates(round_num, active_clients=selected_clients)
        logger.debug(f"Server collected {collected_count} updates.")

        # Server aggregates updates
        server.aggregation()
        logger.debug("Server completed aggregation.")

        # Server updates global model
        server.update_global()
        logger.debug("Server updated global model.")

        # Calculate rejected clients for this round
        current_rejected_count = server.metrics.total_updates_rejected
        rejected_this_round = current_rejected_count - initial_rejected_count # R_t
        accepted_this_round = collected_count - rejected_this_round # A_t
        logger.debug(f"Rejected this round: {rejected_this_round}, Accepted this round: {accepted_this_round}")

        # Retrieve dynamic cost parameters
        alpha = args.alpha_scaling_factor
        beta = args.beta_scaling_factor
        E0 = args.energy_offset

        # S: model size (KB)
        S = model_size_kb

        # N_t: active clients (selected_clients_count)
        N_t = selected_clients_count

        # ρ_t: rejection ratio (from defense pipeline)
        # A_t: anomaly score (from defense pipeline)
        # These are lists, so we need to calculate a representative value for the round.
        # For rejection ratio, we can use the ratio of rejected updates to total selected clients.
        # For anomaly score, we can use the mean of anomaly scores from the defense pipeline.
        
        # Get anomaly scores and filter stats from the server
        round_anomaly_scores = server.last_anomaly_scores
        round_filter_stats = server.last_filter_stats

        # Calculate rejection ratio (ρ_t)
        # If defense pipeline was active and rejected updates, use that. Otherwise, 0.
        total_updates_processed_by_defense = round_filter_stats.get('total_processed', 0)
        updates_rejected_by_defense = round_filter_stats.get('rejected_count', 0)
        
        if total_updates_processed_by_defense > 0:
            rho_t = updates_rejected_by_defense / total_updates_processed_by_defense
        else:
            rho_t = 0.0
        
        # Calculate anomaly score (A_t)
        # Use mean of anomaly scores if available, otherwise 0.
        if round_anomaly_scores:
            A_t = np.mean(round_anomaly_scores)
        else:
            A_t = 0.0

        # Dynamic communication per round (C_t)
        # C_t = S * N_t * (1 - ρ_t) * (1 + α * A_t)
        C_t = S * N_t * (1.0 - rho_t) * (1.0 + alpha * A_t)

        # Dynamic energy per round (E_t)
        # E_t = β * C_t + E_0 * (1 + A_t)
        E_t = beta * C_t + E0 * (1.0 + A_t)

        logger.debug(f"Calculated Comm (KB): {C_t:.2f}, Energy (mJ): {E_t:.2f}")
        logger.debug(f"S: {S:.2f}, N_t: {N_t}, rho_t: {rho_t:.2f}, A_t: {A_t:.2f}")
        logger.debug(f"alpha: {alpha:.2f}, beta: {beta:.2f}, E0: {E0:.2f}")

        # Evaluate global model
        metrics = coordinator.evaluate(server, test_dataset, args, round_num)
        current_accuracy = metrics["test_acc"]
        logger.debug(f"Global model evaluated. Test Acc: {current_accuracy:.4f}")

        round_end_time = time.time() # End timing for the current round
        round_time_duration = round_end_time - round_start_time

        # Store round metrics
        round_metrics = {
            "round": round_num,
            "test_acc": metrics["test_acc"],
            "test_loss": metrics["test_loss"],
            "asr": metrics.get("asr"),
            "asr_loss": metrics.get("asr_loss"),
            "round_time": round_time_duration,
            "active_clients": selected_clients_count,
            "stragglers": None, # Placeholder, needs actual implementation if available
            "dropouts": None, # Placeholder, needs actual implementation if available
            "communication_cost_kb": C_t, # Updated communication cost
            "energy_consumption_mj": E_t, # Updated energy consumption
            "anomaly_score": A_t, # Log the anomaly score
            "rejection_ratio": rho_t, # Log the rejection ratio
            "model_size_kb": S, # Log model size
            "alpha_scaling_factor": alpha, # Log alpha
            "beta_scaling_factor": beta, # Log beta
            "energy_offset": E0 # Log E0
        }
        all_round_metrics.append(round_metrics)

        logger.info(f"Round {round_num:<3}\t"
                   f"Test Acc: {metrics['test_acc']:.4f}\t"
                   f"Test Loss: {metrics['test_loss']:.4f}\t"
                   f"Comm (KB): {C_t:.2f}\t"
                   f"Energy (mJ): {E_t:.2f}\t"
                   f"Anomaly Score: {A_t:.2f}\t"
                   f"Rejection Ratio: {rho_t:.2f}")

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_model_state = copy.deepcopy(server.get_global_model().state_dict())
            # Save best model
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_model_state, output_dir / 'best_global_model.pth')
            logger.info(f"\n🎯 NEW BEST ACCURACY: {best_accuracy:.4f}")

    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING COMPLETE")
    logger.info(f"BEST ACCURACY: {best_accuracy:.4f}")
    logger.info(f"{'='*60}\n")

    # Save all round metrics to a JSON file
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file_path = Path(args.output) / 'training_metrics.json' # Ensure it's directly in args.output
    logger.info(f"Output directory for metrics: {output_dir.resolve()}")
    logger.info(f"Attempting to save metrics to: {metrics_file_path.resolve()}")
    try:
        with open(metrics_file_path, 'w') as f:
            json.dump({"rounds": all_round_metrics, "final_test_accuracy": best_accuracy}, f, indent=4)
        logger.info(f"Metrics saved to {metrics_file_path}")
        logger.info(f"Full path to metrics file: {metrics_file_path.absolute()}")
    except Exception as e:
        logger.error(f"Failed to save training metrics to {metrics_file_path}: {e}", exc_info=True)

    # Load best model if available
    if best_model_state is not None:
        final_model = server.get_global_model()
        final_model.load_state_dict(best_model_state)
    else:
        final_model = server.get_global_model() # Return the last model if no improvement

    logger.info(f"\n🎯 FINAL BEST ACCURACY: {best_accuracy:.4f}")

    # Close all handlers to release the log file
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
