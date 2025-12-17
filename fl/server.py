"""
fl/server.py

Server implementation with:
 - defense pipeline integration (optional)
 - communication + energy logging
 - robust update collection supporting active_clients list
"""

import numpy as np
import logging
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import pickle
import copy # Added for deepcopy

# keep torch import only for vector<->model conversions
import torch

from defenses.pipeline import DefensePipeline  # your project's defense pipeline
from fl.models import get_model
from fl.models.model_utils import model2vec, vec2model, state2vec  # adapt if names differ
from aggregators import get_aggregator


@dataclass
class ServerMetrics:
    rounds_completed: int = 0
    total_updates_received: int = 0
    total_updates_accepted: int = 0
    total_updates_rejected: int = 0
    
    client_accuracies: Dict[int, List[float]] = field(default_factory=lambda: defaultdict(list)) # Per-client accuracy history
    global_model_validation_accuracy: List[float] = field(default_factory=list) # Global model validation accuracy

    aggregation_times: List[float] = field(default_factory=list)
    collection_times: List[float] = field(default_factory=list)

    bytes_per_round: List[int] = field(default_factory=list)    # bytes
    energy_per_round: List[float] = field(default_factory=list)  # mJ

    def to_dict(self) -> Dict[str, Any]:
        return {
            'rounds_completed': self.rounds_completed,
            'total_updates_received': self.total_updates_received,
            'total_updates_accepted': self.total_updates_accepted,
            'total_updates_rejected': self.total_updates_rejected,
            'avg_aggregation_time': float(np.mean(self.aggregation_times)) if self.aggregation_times else 0.0,
            'avg_collection_time': float(np.mean(self.collection_times)) if self.collection_times else 0.0,
            'avg_bytes_per_round': float(np.mean(self.bytes_per_round)) if self.bytes_per_round else 0.0,
            'avg_energy_per_round': float(np.mean(self.energy_per_round)) if self.energy_per_round else 0.0,
            'global_model_validation_accuracy': self.global_model_validation_accuracy,
            'client_accuracies': {cid: accs for cid, accs in self.client_accuracies.items()} # Convert defaultdict to dict
        }


class Server:
    def __init__(self, args, clients: List, test_dataset, train_dataset=None):
        self.args = args
        self.clients = clients
        self.test_dataset = test_dataset
        self.train_dataset = train_dataset

        # Initialize test_loader and criterion
        self.test_loader = self.get_dataloader(test_dataset, train_flag=False)
        self.criterion = torch.nn.CrossEntropyLoss() # Assuming CrossEntropyLoss for classification tasks

        self.logger = logging.getLogger(f"{__name__}.Server")
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter('[Server] %(message)s'))
            self.logger.addHandler(h)
            self.logger.setLevel(logging.INFO)

        # model initialization
        self.global_model = get_model(self.args)
        self.global_weights_vec = model2vec(self.global_model)
        self.previous_weights_vec = None

        # updates storage for current round
        self.updates: List[np.ndarray] = []
        self.update_metadata: List[Dict[str, Any]] = []
        self.participating_clients: List[int] = []

        # metrics
        self.metrics = ServerMetrics()

        # defense pipeline (optional)
        self.defense_pipeline = None
        try:
            # instantiate only if defenses present in args or aggregator needs it
            self.defense_pipeline = DefensePipeline(self.args, logger=self.logger)
            self.logger.info("Defense pipeline initialized successfully")
        except Exception as e:
            # keep going even if defense pipeline not available
            self.logger.debug(f"Defense pipeline not initialized: {e}")
            self.defense_pipeline = None

        # aggregation method (string)
        self.aggregation_method = getattr(self.args, "aggregator", "FedAvg")
        self.logger.info(f"Aggregation method: {self.aggregation_method}")
        self.logger.info(f"Server initialized with {len(self.clients)} clients")

        # energy/comm coefficients
        self.energy_coeff = float(getattr(self.args, "comm_coefficient", getattr(self.args, "energy_coeff", 0.05)))  # mJ per KB
        self.max_update_norm = float(getattr(self.args, "max_update_norm", 1e6))
        self.checkpoint_enabled = getattr(self.args, "save_checkpoints", False)
        self.checkpoint_interval = getattr(self.args, "checkpoint_interval", 10)
        
        # Initialize global model properly
        from fl.models.model_utils import initialize_model_properly
        self.global_model = initialize_model_properly(self.global_model)
        self.global_weights_vec = model2vec(self.global_model)

    # ----------------- collection -----------------
    def collect_updates(self, global_epoch: int, active_clients: Optional[List] = None) -> int:
        """
        Collect updates (client models) from clients provided in active_clients (if None, uses all self.clients).
        Measures collection time and bytes.
        """
        start = time.time()
        self.updates = [] # This will now store client models (or their state_dicts)
        self.update_metadata = []
        self.participating_clients = []
        self.client_data_sizes = [] # Added to store client data sizes for aggregation

        clients_to_collect = active_clients if active_clients is not None else self.clients

        collected = 0
        skipped = 0
        total_bytes = 0
        
        # Store client accuracies for this round
        round_client_accuracies = {}
        self.client_data_sizes = [] # Added to store client data sizes for aggregation

        for client in clients_to_collect:
            # safety: client must have .update produced by client.fetch_updates()
            if not hasattr(client, "model") or client.model is None: # Check for client model directly
                skipped += 1
                continue

            try:
                # Store the client's model (or its state_dict) directly
                client_model_state_dict = client.model.state_dict()
                
                # Estimate size in bytes (vectorize for size estimation)
                arr = state2vec(client_model_state_dict)
                num_bytes = arr.size * 4  # float32 assumption

                norm = float(np.linalg.norm(arr))
                if np.isnan(arr).any() or np.isinf(arr).any():
                    self.logger.warning(f"Client {getattr(client, 'client_id', 'N/A')}: Model parameters contain NaN/Inf -> skipped")
                    skipped += 1
                    continue
                if norm > self.max_update_norm:
                    self.logger.warning(f"Client {getattr(client, 'client_id', 'N/A')}: Model norm too large ({norm:.2e}) -> skipped")
                    skipped += 1
                    continue

                meta = {
                    'client_id': getattr(client, 'client_id', None),
                    'norm': norm,
                    'num_bytes': int(num_bytes),
                    'num_samples': len(client.train_loader.dataset) # Use actual dataset size
                }
                
                # Collect client-level accuracy
                if hasattr(client, 'client_test_accuracy'):
                    round_client_accuracies[client.client_id] = client.client_test_accuracy
                    self.metrics.client_accuracies[client.client_id].append(client.client_test_accuracy)

                self.updates.append(client_model_state_dict) # Store state_dict
                self.update_metadata.append(meta)
                self.participating_clients.append(meta['client_id'])
                self.client_data_sizes.append(len(client.train_loader.dataset)) # Store client data size
                collected += 1
                total_bytes += num_bytes

            except Exception as e:
                self.logger.warning(f"Failed to collect from client {getattr(client, 'client_id', 'N/A')}: {e}")
                skipped += 1
                continue
        
        if round_client_accuracies:
            avg_round_acc = np.mean(list(round_client_accuracies.values()))
            self.logger.info(f"  Avg client local accuracy: {avg_round_acc:.4f}")
            
        
        # Model validation phase before aggregation
        self.logger.info("\n[Validation Phase]")
        global_model_copy = get_model(self.args)
        vec2model(self.global_weights_vec, global_model_copy)
        val_acc, val_loss = self.test(global_model_copy, self.test_loader)
        self.metrics.global_model_validation_accuracy.append(val_acc)
        self.logger.info(f"  Global model validation accuracy: {val_acc:.4f}, Loss: {val_loss:.4f}")
        
        # If defense pipeline is active, it might use this validation accuracy to adjust thresholds
        if self.defense_pipeline and hasattr(self.defense_pipeline, 'adapt_to_validation_accuracy'):
            self.defense_pipeline.adapt_to_validation_accuracy(val_acc)
            

        elapsed = time.time() - start
        self.metrics.total_updates_received += collected
        self.metrics.collection_times.append(elapsed)
        self.metrics.bytes_per_round.append(total_bytes)
        self.logger.info("Collection Summary:")
        self.logger.info(f"  ✓ Collected: {collected} updates")
        if skipped > 0:
            self.logger.info(f"  ⊗ Skipped:   {skipped} clients")
        self.logger.info(f"  Time:       {elapsed:.3f}s")
        self.logger.info(f"  Data:       {total_bytes/1024.0:.2f} KB")

        return collected

    # ----------------- aggregation -----------------
    def aggregation(self):
        if not self.updates:
            self.logger.warning("No updates to aggregate")
            return

        start = time.time()
        self.previous_weights_vec = self.global_weights_vec.copy() if self.global_weights_vec is not None else None

        # Apply defense pipeline if present; expected to return list of state_dicts
        sanitized_state_dicts = self.updates
        if self.defense_pipeline:
            try:
                # The defense pipeline expects numpy vectors, so convert state_dicts to vectors for sanitization
                updates_as_vectors = [state2vec(sd) for sd in self.updates]
                sanitized_vectors = self.defense_pipeline.sanitize_updates(updates_as_vectors, client_ids=self.participating_clients, global_model=self.global_weights_vec)
                
                # Filter Nones and convert back to state_dicts
                sanitized_state_dicts = []
                for vec in sanitized_vectors:
                    if vec is not None:
                        temp_model = get_model(self.args)
                        vec2model(vec, temp_model)
                        sanitized_state_dicts.append(temp_model.state_dict())
                
                rejected = len(self.updates) - len(sanitized_state_dicts)
                if rejected > 0:
                    self.metrics.total_updates_rejected += rejected
                    self.logger.info(f"  Defense rejected: {rejected}/{len(self.updates)} updates")
                if not sanitized_state_dicts:
                    self.logger.warning("All updates rejected by defense; falling back to raw updates")
                    sanitized_state_dicts = self.updates
            except Exception as e:
                self.logger.warning(f"Defense pipeline failed: {e}")
                sanitized_state_dicts = self.updates

        # choose aggregator
        aggregated_model = None
        try:
            if self.aggregation_method in ("FedAvg", "FedAvgWeight", "SystemAware", None): # Added SystemAware
                # Pass global_model (PyTorch object), client state_dicts, and client data sizes
                aggregated_model = self._fedavg_aggregate(self.global_model, sanitized_state_dicts, self.client_data_sizes)
            elif self.aggregation_method == "Median":
                # Convert state_dicts to vectors for median aggregation
                updates_as_vectors = [state2vec(sd) for sd in sanitized_state_dicts]
                agg_vec = self._median(updates_as_vectors)
                aggregated_model = get_model(self.args)
                vec2model(agg_vec, aggregated_model)
            elif self.aggregation_method == "TrimmedMean":
                # Convert state_dicts to vectors for trimmed mean aggregation
                updates_as_vectors = [state2vec(sd) for sd in sanitized_state_dicts]
                agg_vec = self._trimmed_mean(updates_as_vectors)
                aggregated_model = get_model(self.args)
                vec2model(agg_vec, aggregated_model)
            else:
                try:
                    # Attempt to get aggregator from registry
                    aggregator_class = get_aggregator(self.aggregation_method)
                    aggregator_instance = aggregator_class(self.args) # Assuming aggregator takes args
                    # Aggregators from registry typically expect vectors
                    updates_as_vectors = [state2vec(sd) for sd in sanitized_state_dicts]
                    agg_vec = aggregator_instance.aggregate(updates_as_vectors)
                    aggregated_model = get_model(self.args)
                    vec2model(agg_vec, aggregated_model)
                except KeyError:
                    self.logger.warning(f"Unknown aggregator '{self.aggregation_method}', using FedAvg")
                    aggregated_model = self._fedavg_aggregate(self.global_model, sanitized_state_dicts, self.client_data_sizes) # Fallback to improved FedAvg
        except Exception as e:
            self.logger.error(f"Aggregation failed: {e}")
            raise

        agg_time = time.time() - start
        self.metrics.aggregation_times.append(agg_time)
        self.metrics.total_updates_accepted += len(sanitized_state_dicts)

        # set new global_weights_vec from aggregated_model
        self.global_weights_vec = model2vec(aggregated_model)
        self._log_aggregation_summary([state2vec(sd) for sd in sanitized_state_dicts], agg_time) # Log with vectors

        # log energy/communication for this round
        self._log_energy_comm()

    def _fedavg_aggregate(self, global_model, client_state_dicts, client_weights):
        """
        Weighted FedAvg aggregation, returns a PyTorch model.
        client_weights: number of samples per client
        """
        global_dict = global_model.state_dict()
        
        # Initialize with zeros
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
        
        # Weighted sum
        total_weight = sum(client_weights)
        
        for client_state, weight in zip(client_state_dicts, client_weights):
            for key in global_dict.keys():
                global_dict[key] += (client_state[key].float() * weight / total_weight)
        
        global_model.load_state_dict(global_dict)
        return global_model

    def _fedavg(self, updates: List[np.ndarray]) -> np.ndarray:
        # This is the old _fedavg, kept for compatibility if needed by other aggregators
        return np.mean(np.stack(updates, axis=0), axis=0)

    def _median(self, updates: List[np.ndarray]) -> np.ndarray:
        return np.median(np.stack(updates, axis=0), axis=0)

    def _trimmed_mean(self, updates: List[np.ndarray]) -> np.ndarray:
        stacked = np.stack(updates, axis=0)
        trim_ratio = float(getattr(self.args, "trim_ratio", 0.1))
        trim_count = max(1, int(len(updates) * trim_ratio))
        sorted_updates = np.sort(stacked, axis=0)
        trimmed = sorted_updates[trim_count:-trim_count] if trimmed.shape[0] > 0 else stacked
        return np.mean(trimmed, axis=0)

    def _log_aggregation_summary(self, updates: List[np.ndarray], agg_time: float):
        self.logger.info("\nAggregation Summary:")
        self.logger.info(f"  Method: {self.aggregation_method}")
        self.logger.info(f"  Updates aggregated: {len(updates)}")
        self.logger.info(f"  Time: {agg_time:.3f}s")
        norms = [float(np.linalg.norm(u)) for u in updates]
        self.logger.info("  Update norms:")
        self.logger.info(f"    Mean: {np.mean(norms):.4f}")
        self.logger.info(f"    Std:  {np.std(norms):.4f}")
        self.logger.info(f"    Min:  {np.min(norms):.4f}")
        self.logger.info(f"    Max:  {np.max(norms):.4f}")

        if self.previous_weights_vec is not None and self.global_weights_vec is not None:
            change = float(np.linalg.norm(self.global_weights_vec - self.previous_weights_vec))
            self.logger.info(f"  Model change magnitude: {change:.4f}")

    def _log_energy_comm(self):
        """Compute and log energy + comm metrics for last collection"""
        if not self.metrics.bytes_per_round:
            return

        last_bytes = int(self.metrics.bytes_per_round[-1])
        kb = last_bytes / 1024.0
        energy_mj = kb * self.energy_coeff
        self.metrics.energy_per_round.append(energy_mj)
        self.logger.info(f"\n[Metrics] Round {self.metrics.rounds_completed}:")
        self.logger.info(f"  Communication: {kb:.2f} KB")
        self.logger.info(f"  Energy:        {energy_mj:.3f} mJ (coeff={self.energy_coeff})")

    # ----------------- apply weights to PyTorch model -----------------
    def update_global(self):
        if self.global_weights_vec is None:
            self.logger.warning("No global weights set, skipping update")
            return

        # safety: convert to torch tensor then apply to self.global_model parameters
        try:
            vec_t = torch.tensor(self.global_weights_vec, dtype=torch.float32)
            torch.nn.utils.vector_to_parameters(vec_t.to(next(self.global_model.parameters()).device), self.global_model.parameters())
        except Exception:
            # fallback: do nothing if conversion fails
            pass

        self.metrics.rounds_completed += 1
        self.logger.info("\n✅ Global model updated successfully")

        # checkpoint if needed
        if self.checkpoint_enabled and self.metrics.rounds_completed % self.checkpoint_interval == 0:
            # Call save_model with a default path if checkpointing is enabled
            cp_dir = Path(self.args.output) / "server_checkpoints"
            cp_dir.mkdir(parents=True, exist_ok=True)
            cp_file = cp_dir / f"round_{self.metrics.rounds_completed}.pkl"
            self.save_model(cp_file)

    def save_model(self, checkpoint_path: Path):
        """
        Saves the current global model and server metrics to a specified path.
        """
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({
                    'round': self.metrics.rounds_completed,
                    'global_weights': self.global_weights_vec,
                    'metrics': self.metrics.to_dict()
                }, f)
            self.logger.info(f"Model checkpoint saved to: {checkpoint_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save model checkpoint to {checkpoint_path}: {e}")

    # ----------------- utils -----------------
    def _flatten_state_dict_to_vector(self, state_dict: Dict[str, Any]) -> np.ndarray:
        """
        Flatten a state_dict (mapping keys->torch.Tensor) into 1D numpy array.
        This is a simple deterministic ordering (sorted keys).
        """
        parts = []
        for k in sorted(state_dict.keys()):
            v = state_dict[k]
            if isinstance(v, torch.Tensor):
                parts.append(v.cpu().numpy().ravel())
            else:
                parts.append(np.array(v).ravel())
        return np.concatenate(parts).astype(np.float32)

    def get_global_model(self):
        return self.global_model

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.to_dict()

    def get_dataloader(self, dataset, train_flag):
        # Assuming args has batch_size and num_workers
        batch_size = self.args.batch_size
        num_workers = getattr(self.args, "num_workers", 0) # Default to 0 if not specified

        if train_flag:
            return torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
            )
        else:
            return torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
            )

    def criterion_fn(self, y_pred, y_true, **kwargs):
        return torch.nn.CrossEntropyLoss()(y_pred, y_true)

    def test(self, model, test_loader, imbalanced=False):
        model.eval()
        tail_cls_from = self.args.tail_cls_from if imbalanced else 0
        overall_correct, test_loss, num_samples = 0, 0, 0
        rest_correct, rest_samples = 0, 0

        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(
                    self.args.device), targets.to(self.args.device)
                pred_probs = model(images)
                loss = self.criterion_fn(pred_probs, targets)
                predicted = torch.argmax(pred_probs.data, 1)
                num_samples += len(targets)
                overall_correct += (predicted == targets).sum().item()
                test_loss += loss.item()

                if imbalanced:
                    # calculate the rest class accuracy, from 5-9 for 10 classes
                    rest_mask = targets >= tail_cls_from
                    rest_correct += (predicted[rest_mask]
                                     == targets[rest_mask]).sum().item()
                    rest_samples += rest_mask.sum().item()

        overall_accuracy = overall_correct / num_samples
        test_loss /= len(test_loader) # Changed from num_samples to len(test_loader) for consistency

        if imbalanced:
            rest_accuracy = rest_correct / rest_samples if rest_samples > 0 else 0
            return overall_accuracy, rest_accuracy, test_loss

        return overall_accuracy, test_loss

    def __repr__(self):
        return f"Server(clients={len(self.clients)}, rounds={self.metrics.rounds_completed}, agg={self.aggregation_method})"
