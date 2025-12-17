import logging
import time
import gc
import torch
import numpy as np

from .server import Server
from .coordinator import init_clients, set_fl_algorithm, evaluate
from global_utils import avg_value
from datapreprocessor.data_utils import load_data, split_dataset
from defenses.pipeline import DefensePipeline
from fl.models import get_model # Import get_model

# The omniscient_attack function, moved from main.py
def omniscient_attack(clients, server, global_epoch):
    # existing project provides attackers; keep a safe no-op if missing
    attackers = [c for c in clients if getattr(c, 'category', None) == 'attacker']
    if not attackers:
        return
    if hasattr(attackers[0], 'omniscient'):
        malicious_updates = attackers[0].omniscient(clients)
        if malicious_updates is None:
            return
        single_update = (len(malicious_updates.shape) == 1 or malicious_updates.shape[0] == 1)
        if single_update:
            for a in attackers:
                a.update = malicious_updates
        else:
            for a, u in zip(attackers, malicious_updates):
                a.update = u

class SimpleOrchestrator:
    def __init__(self, args):
        self.args = args
        self.logger = args.logger if hasattr(args, 'logger') else logging.getLogger(__name__)
        self.server = None
        self.clients = []
        self.train_dataset = None
        self.test_dataset = None
        self.global_model = None # Initialize global_model
        self.global_weights_vec = None # Will be derived from global_model in Server

        self.avg_train_loss = []
        self.avg_train_acc = []
        self.epoch_msg = ""

    def setup(self):
        self.logger.info("Setting up orchestrator...")

        # Load datasets
        self.train_dataset, self.test_dataset = load_data(self.args)

        # Split dataset indices across clients
        client_indices, self.test_dataset = split_dataset(self.args, self.train_dataset, self.test_dataset)

        self.clients = init_clients(self.args, client_indices, self.train_dataset, self.test_dataset)

        # Initialize global model
        self.global_model = get_model(self.args)

        # Initialize defense pipeline if a defense is specified
        if hasattr(self.args, 'defense') and self.args.defense != 'NoDefense':
            # Ensure verbose flag is passed to defense_params if set
            if not hasattr(self.args, 'defense_params') or self.args.defense_params is None:
                self.args.defense_params = {}
            if hasattr(self.args, 'verbose') and self.args.verbose:
                self.args.defense_params['verbose'] = True
            self.args.defense_pipeline = DefensePipeline(self.args, self.logger)
        else:
            self.args.defense_pipeline = None

        # Pass the actual global_model, clients, and datasets to the Server
        self.server = Server(self.args, self.global_model, self.clients, self.test_dataset, self.train_dataset)
        self.global_weights_vec = self.server.global_weights_vec # Update orchestrator's global_weights_vec

        set_fl_algorithm(self.args, self.server, self.clients)
        self.logger.info("Orchestrator setup complete.")

    def run(self):
        self.logger.info("Starting federated learning training...")
        start_time = time.time()

        for global_epoch in range(self.args.epochs): # Assuming args.epochs defines total epochs
            self.logger.info(f"--- Global Epoch {global_epoch + 1}/{self.args.epochs} ---")
            self.epoch_msg = f"Epoch {global_epoch + 1}:"

            self.avg_train_loss = []
            self.avg_train_acc = []

            active_clients = self.clients

            # Clients perform local training and fetch updates
            for client in active_clients:
                # Load the current global model onto the client
                client.load_global_model(self.server.global_weights_vec)
                
                # Client performs local training and prepares its update
                client.fetch_updates()

                # Collect metrics from client after training
                # Assuming client.local_training or fetch_updates updates these
                # For now, we'll use dummy values if actual metrics aren't readily available
                # The client.local_training method returns avg_acc, avg_loss
                # We need to ensure these are accessible or passed back.
                # For now, let's assume client.fetch_updates implicitly updates client.avg_train_loss and client.avg_train_acc
                # If not, we'll need to adjust client.fetch_updates to return them or store them on the client object.
                # For now, we'll use placeholder values if not available.
                self.avg_train_loss.append(getattr(client, 'avg_local_loss', 0.1))
                self.avg_train_acc.append(getattr(client, 'avg_local_acc', 0.9))

            # Server side: collect updates (trained models) from the selected clients
            self.server.collect_updates(global_epoch, active_clients)


            _avg_train_loss_val = avg_value(self.avg_train_loss) if self.avg_train_loss else 0.0
            _avg_train_acc_val = avg_value(self.avg_train_acc) if self.avg_train_acc else 0.0
            self.epoch_msg += f"\tTrain Acc: {_avg_train_acc_val:.4f}\tTrain loss: {_avg_train_loss_val:.4f}\t"

            try:
                omniscient_attack(self.clients, self.server, global_epoch)
            except Exception as e:
                self.logger.warning(f"Omniscient attack failed: {e}")
                pass

            self.server.aggregation()
            self.server.update_global()

            test_stats = evaluate(self.server, self.test_dataset, self.args, global_epoch)
            self.epoch_msg += "\t".join([f"{key}: {value:.4f}" for key, value in test_stats.items()])
            self.logger.info(self.epoch_msg)

            gc.collect()

        end_time = time.time()
        minutes, seconds = divmod(int(end_time - start_time), 60)
        self.logger.info(f"Training finished on {time.asctime(time.localtime(end_time))} using {minutes} minutes and {seconds} seconds in total.")
        self.logger.info("Federated learning training finished.")
