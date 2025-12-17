"""
main.py
Federated learning orchestrator / CLI entrypoint.
Includes:
 - partial participation
 - straggler simulation
 - device heterogeneity (per-client LR)
 - attack hook (omniscient/adaptive)
 - metrics collection (communication, energy, round time)
"""





import sys
import os
import datetime     # <-- add this

import gc
import logging
import time
import random
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from fl import coordinator
from global_args import benchmark_preprocess, read_args, override_args, single_preprocess
from global_utils import avg_value, print_filtered_args, setup_logger, setup_seed
from datapreprocessor.data_utils import load_data, split_dataset
from fl.server import Server
from plot_utils import plot_accuracy




class ConsoleLogger:
    """
    Mirror everything written to stdout/stderr into a log file,
    while still showing it on the console.
    """
    def __init__(self, log_dir="logs", prefix="fl_training"):
        os.makedirs(log_dir, exist_ok=True)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{prefix}_{ts}.log")

        # line-buffered text file
        self.log_file = open(self.log_path, "w", buffering=1, encoding="utf-8")

        # keep original streams
        self._stdout = sys.stdout
        self._stderr = sys.stderr

        # redirect stdout/stderr to this object
        sys.stdout = self
        sys.stderr = self

    def write(self, text: str):
        # forward to real console
        self._stdout.write(text)
        # save to file
        self.log_file.write(text)

    def flush(self):
        self._stdout.flush()
        self.log_file.flush()

    def close(self):
        # restore original streams
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self.log_file.close()




class TrainingMetrics:
    def __init__(self):
        self.train_accuracies: List[float] = []
        self.train_losses: List[float] = []
        self.test_accuracies: List[float] = []
        self.test_losses: List[float] = []
        self.round_times: List[float] = []
        self.client_participation: List[int] = []
        self.straggler_counts: List[int] = []
        self.energy_per_round: List[float] = []
        self.comm_kb_per_round: List[float] = []
        self.attack_success: List[bool] = []

    def to_dict(self):
        return {
            'train_accuracies': self.train_accuracies,
            'train_losses': self.train_losses,
            'test_accuracies': self.test_accuracies,
            'test_losses': self.test_losses,
            'round_times': self.round_times,
            'client_participation': self.client_participation,
            'straggler_counts': self.straggler_counts,
            'energy_per_round': self.energy_per_round,
            'comm_kb_per_round': self.comm_kb_per_round,
            'attack_success': self.attack_success
        }

    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def omniscient_attack(clients: List, server: Server, global_epoch: int) -> bool:
    """
    Execute omniscient attack if configured.
    Returns True if an attacker injected updates this round.
    """
    attackers = [c for c in clients if getattr(c, "category", "") == "attacker" and "omniscient" in getattr(c, "attributes", [])]
    if not attackers:
        return False

    # Allow attackers to adapt (if they implement adaptive_round)
    for a in attackers:
        if hasattr(a, "adaptive_round"):
            try:
                a.adaptive_round(global_epoch, server, clients)
            except Exception as e:
                # fail gracefully
                if hasattr(server, "logger"):
                    server.logger.debug(f"[AdaptiveAttack] adaptive_round failed: {e}")

    # generate malicious updates using first attacker's omniscient method (convention in this codebase)
    malicious_updates = attackers[0].omniscient(clients)
    if malicious_updates is None:
        return False

    # distribute updates (either single update or one per attacker)
    try:
        single_update = (len(getattr(malicious_updates, "shape", ())) == 1 or (getattr(malicious_updates, "shape", (None,))[0] == 1))
    except Exception:
        single_update = False

    if single_update:
        for a in attackers:
            a.update = malicious_updates
    else:
        for a, upd in zip(attackers, malicious_updates):
            a.update = upd

    return True


class FederatedLearningOrchestrator:
    def __init__(self, args):
        self.args = args
        self.logger = args.logger
        self.metrics = TrainingMetrics()

        # flags and params
        self.partial_participation_enabled = getattr(args, "partial_participation", False) or (getattr(args, "participation_rate", 1.0) < 1.0)
        self.participation_rate = float(getattr(args, "participation_rate", 1.0))
        self.simulate_stragglers = getattr(args, "simulate_stragglers", False)
        self.straggler_probability = float(getattr(args, "straggler_probability", getattr(args, "straggler_prob", 0.0)))
        # allow either 'straggler_delay' or 'straggler_mean_delay'
        self.straggler_delay = float(getattr(args, "straggler_delay", getattr(args, "straggler_mean_delay", 2.0)))
        self.straggler_timeout = float(getattr(args, "straggler_timeout", getattr(args, "timeout_threshold", 10.0)))

        # heterogeneity flags
        self.simulate_heterogeneity = getattr(args, "simulate_heterogeneity", False) or getattr(args, "heterogeneous_clients", False)
        self.hetero_lr_min = float(getattr(args, "hetero_lr_min", getattr(args, "hetero_lr_low", 0.005)))
        self.hetero_lr_max = float(getattr(args, "hetero_lr_max", getattr(args, "hetero_lr_high", 0.05)))

        # components (created during setup)
        self.clients = []
        self.server: Optional[Server] = None
        self.train_dataset = None
        self.test_dataset = None
        self.client_indices = None

        self.start_time = None

        # Early stopping parameters
        self.early_stopping_patience = getattr(args, "early_stopping_patience", 10) # Default patience
        self.best_test_acc = -float('inf')
        self.epochs_no_improve = 0
        self.best_global_epoch = -1

    def setup(self):
        self.logger.info("=" * 70)
        self.logger.info("Federated Learning setup")
        self.logger.info("=" * 70)
        self.start_time = time.time()

        # fix randomness
        setup_seed(self.args.seed)
        self.logger.info(f"Random seed: {self.args.seed}")

        # load & split
        self.train_dataset, self.test_dataset = load_data(self.args)
        self.client_indices = split_dataset(self.train_dataset, self.args.num_clients, self.args.distribution, self.args.num_shards)
        self.logger.info(f"Data partitioned into {len(self.client_indices)} clients")

        # init clients and server using coordinator (keeps client creation consistent with project)
        self.clients = coordinator.init_clients(self.args, self.client_indices, self.train_dataset, self.test_dataset)
        self.server = Server(self.args, self.clients, self.test_dataset, self.train_dataset)

        # set FL algorithm handlers
        coordinator.set_fl_algorithm(self.args, self.server, self.clients)

        # log feature flags
        self.logger.info(f"[Server] Partial participation enabled: {self.partial_participation_enabled} (rate={self.participation_rate})")
        if self.simulate_stragglers:
            self.logger.info(f"[Server] Straggler simulation enabled: prob={self.straggler_probability}, delay_mean={self.straggler_delay}s, timeout={self.straggler_timeout}s")
        if self.simulate_heterogeneity:
            self.logger.info(f"[Server] Heterogeneity enabled: lr_range=[{self.hetero_lr_min}, {self.hetero_lr_max}]")

    def _select_clients(self, round_idx: int) -> List:
        if not self.partial_participation_enabled:
            return self.clients

        num_clients = max(1, int(len(self.clients) * self.participation_rate))
        # simple random selection
        selected = random.sample(self.clients, num_clients)
        return selected

    def _apply_heterogeneity(self, client) -> Dict[str, Any]:
        """
        If heterogeneity is enabled, adjust per-client learning rate (and optionally other settings).
        Returns a dict of original values to restore later.
        """
        original = {}
        if not self.simulate_heterogeneity:
            return original

        # some clients may not expose learning_rate attribute; adapt gracefully
        base_lr = getattr(self.args, "learning_rate", 0.01)
        lr = random.uniform(self.hetero_lr_min, self.hetero_lr_max)
        if hasattr(client, "optimizer"):
            try:
                # record original optimizer param groups
                original['optimizer_state'] = None  # placeholder: we won't deep-copy optimizer state here
                # set lr for optimizer param groups
                for g in client.optimizer.param_groups:
                    g['lr'] = lr
            except Exception:
                pass

        # also set a property on client for easier logging
        setattr(client, "_hetero_lr", lr)
        self.logger.info(f"[Client {getattr(client, 'client_id', 'N/A')}] Using heterogeneous LR: {lr:.5f}")
        return original

    def _restore_client(self, client, original: Dict[str, Any]):
        # restore any changed settings (we only set lr above)
        if hasattr(client, "optimizer") and '_hetero_lr' in client.__dict__:
            # restore to global learning rate
            global_lr = getattr(self.args, "learning_rate", 0.01)
            for g in client.optimizer.param_groups:
                g['lr'] = global_lr
            del client.__dict__['_hetero_lr']

    def _simulate_straggler_delay(self, client) -> Tuple[bool, float]:
        if not self.simulate_stragglers:
            return False, 0.0
        if random.random() < self.straggler_probability:
            delay = random.uniform(0.5, self.straggler_delay)
            # cap by timeout
            if delay > self.straggler_timeout:
                delay = self.straggler_timeout
            return True, delay
        return False, 0.0

    def train(self):
        if self.server is None or not self.clients:
            raise RuntimeError("Server and clients must be initialized via setup() before train()")

        # global weights vector comes from server
        for global_epoch in range(self.args.epochs):
            round_start = time.time()
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Round {global_epoch}")
            self.logger.info(f"{'='*60}")

            # select participants
            active_clients = self._select_clients(global_epoch)
            self.logger.info(f"[Participation] Selected {len(active_clients)}/{len(self.clients)} clients")

            avg_train_acc = []
            avg_train_loss = []
            straggler_count = 0

            for client in active_clients:
                # dropout simulation
                dropout_prob = getattr(self.args, "dropout_probability", 0.0)
                if random.random() < dropout_prob:
                    self.logger.debug(f"[Dropout] Client {client.client_id} dropped this round")
                    continue

                orig = self._apply_heterogeneity(client)

                # load global model into client
                try:
                    client.load_global_model(self.server.global_weights_vec)
                except Exception:
                    # some clients may accept vector directly
                    try:
                        client.load_global_model(self.server.get_global_model())
                    except Exception:
                        pass

                # simulate straggler
                is_straggler, delay = self._simulate_straggler_delay(client)
                if is_straggler:
                    straggler_count += 1
                    self.logger.info(f"[Straggler] Client {client.client_id} delayed by {delay:.2f}s")
                    time.sleep(delay)

                # run local training
                try:
                    train_acc, train_loss = client.local_training()
                except Exception as e:
                    self.logger.warning(f"[Client {getattr(client, 'client_id', 'N/A')}] training failed: {e}")
                    train_acc, train_loss = 0.0, 0.0

                try:
                    client.fetch_updates()
                except Exception as e:
                    self.logger.warning(f"[Client {getattr(client, 'client_id', 'N/A')}] fetch_updates failed: {e}")

                avg_train_acc.append(train_acc)
                avg_train_loss.append(train_loss)

                # restore client settings if needed
                if orig:
                    self._restore_client(client, orig)

            # Collect updates from all active clients first
            # This is necessary for adaptive attacks that modify existing updates
            # rather than generating them from scratch.
            collected_updates_from_active_clients = []
            for client in active_clients:
                if hasattr(client, "update") and client.update is not None:
                    collected_updates_from_active_clients.append(client.update)

            # Separate benign and malicious clients
            benign_clients = [c for c in active_clients if getattr(c, "category", "") == "benign"]
            adaptive_attack_clients = [c for c in active_clients if getattr(c, "category", "") == "attacker" and self.args.attack == "AdaptiveAttack"]

            attack_happened = False
            if adaptive_attack_clients:
                # Assuming AdaptiveAttack modifies existing updates,
                # we need to pass the benign updates to its generate method.
                # For simplicity, we'll collect all updates and then let the adaptive attack modify them.
                # In a more complex scenario, you might need to explicitly separate benign updates.
                benign_updates_for_attack = [c.update for c in benign_clients if hasattr(c, "update") and c.update is not None]
                
                if benign_updates_for_attack:
                    # The AdaptiveAttack expects a list of torch.Tensors
                    benign_updates_tensor = [torch.tensor(u) for u in benign_updates_for_attack]
                    
                    # Call the generate method of the first adaptive attacker (convention)
                    # This assumes all adaptive attackers use the same logic and parameters
                    malicious_updates_tensor = adaptive_attack_clients[0].generate(benign_updates_tensor)

                    # Distribute the malicious updates back to the adaptive attack clients
                    for i, attacker_client in enumerate(adaptive_attack_clients):
                        if i < len(malicious_updates_tensor):
                            attacker_client.update = malicious_updates_tensor[i].cpu().numpy().flatten()
                            attack_happened = True
                        else:
                            self.logger.warning(f"Not enough malicious updates generated for client {attacker_client.client_id}")
                    if attack_happened:
                        self.logger.info("[Attack] Adaptive malicious updates injected this round")
                else:
                    self.logger.warning("[Attack] No benign updates available for AdaptiveAttack to generate malicious updates.")
            else:
                # If no AdaptiveAttack, check for omniscient attacks
                try:
                    attack_happened = omniscient_attack(self.clients, self.server, global_epoch)
                    if attack_happened:
                        self.logger.info("[Attack] Malicious updates injected this round")
                except Exception as e:
                    self.logger.debug(f"[Attack] omniscient_attack failed or not active: {e}")
                    attack_happened = False

            # server collects updates from participating clients (server will handle which updates are present on clients)
            # The active_clients list now contains clients with their potentially modified updates
            collected_count = self.server.collect_updates(global_epoch, active_clients)
            self.server.aggregation()
            self.server.update_global()

            # evaluate using coordinator's evaluate (keeps compatibility)
            test_stats = coordinator.evaluate(self.server, self.test_dataset, self.args, global_epoch)

            # record per-round metrics
            round_time = time.time() - round_start
            self.metrics.train_accuracies.append(avg_value(avg_train_acc))
            self.metrics.train_losses.append(avg_value(avg_train_loss))
            self.metrics.test_accuracies.append(test_stats.get("test_acc", 0.0))
            self.metrics.test_losses.append(test_stats.get("test_loss", 0.0))
            self.metrics.round_times.append(round_time)
            self.metrics.client_participation.append(collected_count)
            self.metrics.straggler_counts.append(straggler_count)
            # get the most recent energy/comm from server (server keeps bytes/energy per round)
            if hasattr(self.server, "metrics"):
                if self.server.metrics.energy_per_round:
                    self.metrics.energy_per_round.append(self.server.metrics.energy_per_round[-1])
                if self.server.metrics.bytes_per_round:
                    kb = self.server.metrics.bytes_per_round[-1] / 1024.0
                    self.metrics.comm_kb_per_round.append(kb)

            self.metrics.attack_success.append(bool(attack_happened))

            # log summary
            msg = (f"Epoch {global_epoch:<3}\t"
                   f"Train Acc: {avg_value(avg_train_acc):.4f}\t"
                   f"Train loss: {avg_value(avg_train_loss):.4f}\t"
                   f"Test Acc: {test_stats.get('test_acc', 0.0):.4f}\t"
                   f"Test Loss: {test_stats.get('test_loss', 0.0):.4f}")
            self.logger.info(msg)

            gc.collect()

            # Early stopping logic
            current_test_acc = test_stats.get('test_acc', 0.0)
            if current_test_acc > self.best_test_acc:
                self.best_test_acc = current_test_acc
                self.epochs_no_improve = 0
                self.best_global_epoch = global_epoch
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered at epoch {global_epoch}. Best test accuracy: {self.best_test_acc:.4f} at epoch {self.best_global_epoch}.")
                    break # Exit the training loop

        # finalize: save metrics and plots
        out_dir = Path(self.args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = out_dir / "training_metrics.json"
        self.metrics.save(metrics_path)
        try:
            plot_accuracy(self.args.output)
        except Exception:
            self.logger.debug("plot_accuracy failed (maybe missing matplotlib)")

        total_time = time.time() - self.start_time
        self.logger.info(f"Training finished. Total time: {total_time:.2f}s")


def main(args, cli_args):
    # preprocess args
    if cli_args.benchmark:
        benchmark_preprocess(args)
    else:
        override_args(args, cli_args)
        single_preprocess(args)

    # setup logger
    args.logger = setup_logger(__name__, f'{args.output}', level=logging.INFO)
    print_filtered_args(args, args.logger)

    orchestrator = FederatedLearningOrchestrator(args)
    orchestrator.setup()
    orchestrator.train()


# if __name__ == "__main__":
#     args, cli_args = read_args()
#     main(args, cli_args)



if __name__ == "__main__":
    # start capturing console -> log file
    console_logger = ConsoleLogger(log_dir="logs", prefix="fl_training")

    try:
        args, cli_args = read_args()
        main(args, cli_args)
    finally:
        # stop capturing and close file
        console_logger.close()
        print(f"Logs saved to: {console_logger.log_path}")
