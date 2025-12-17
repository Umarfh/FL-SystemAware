# aggregators/systemaware.py
"""
SystemAware aggregator: combines similarity, reputation and system metrics
(energy, data size) to weight client updates robustly.
This module expects `updates` to be a list of either numpy arrays or dict state_dicts.
If provided as (update, metadata) tuples, it will read metadata keys:
- 'client_id', 'reputation', 'energy', 'data_size', 'sim'
"""
import numpy as np
import torch
import logging
from aggregators.aggregatorbase import AggregatorBase
from aggregators import aggregator_registry

logger = logging.getLogger(__name__)

@aggregator_registry.register("SystemAware", "aggregator")
class SystemAware(AggregatorBase):
    """
    Aggregator that:
    - Accepts either list[np.ndarray] or list[(np.ndarray, meta_dict)]
    - Computes cosine similarity to mean, uses reputation, energy/data_size to weight
    - Returns aggregated vector (numpy) or state_dict if inputs were state_dicts.
    """
    def __init__(self, args, train_dataset=None, **kwargs):
        super().__init__(args)
        self.args = args
        self.alpha = float(getattr(args, "systemaware_alpha", 1.0))
        self.sim_threshold = float(getattr(args, "systemaware_sim_threshold", 0.0))
        self.logger = logger

    def aggregate(self, updates, last_global_model=None, global_weights_vec=None,
                  global_epoch=None, client_ids=None, **kwargs):
        """
        Entry point called by server. Accept flexible input formats.
        """
        # Accept both dict(client_id->update) and list
        updates_list = []
        metas = []

        # If updates is dict-like (server may pass dict)
        if isinstance(updates, dict):
            for cid, upd in updates.items():
                updates_list.append(upd)
                metas.append({'client_id': cid})
        elif isinstance(updates, list):
            for item in updates:
                if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[1], dict):
                    updates_list.append(item[0])
                    metas.append(item[1])
                else:
                    updates_list.append(item)
                    metas.append({})
        else:
            raise ValueError("Unsupported updates type for SystemAware")

        # Detect original format (state_dict or vector)
        original_format = None
        state_ref = None
        flat_updates = []
        for u in updates_list:
            if isinstance(u, dict):
                if original_format is None:
                    original_format = 'state_dict'
                    state_ref = u
                flat_updates.append(self._flatten_state_dict(u))
            elif isinstance(u, torch.Tensor):
                if original_format is None:
                    original_format = 'tensor'
                flat_updates.append(u.cpu().numpy().flatten())
            else:
                if original_format is None:
                    original_format = 'numpy'
                flat_updates.append(np.array(u).flatten())

        if len(flat_updates) == 0:
            raise ValueError("No updates provided")

        arr = np.stack(flat_updates, axis=0)  # shape (n_clients, dim)

        # compute mean update
        mean_update = np.mean(arr, axis=0)
        # cosine similarities
        def cos_sim(a, b):
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na < 1e-9 or nb < 1e-9:
                return 0.0
            return float(np.dot(a, b) / (na * nb))

        sims = np.array([cos_sim(row, mean_update) for row in arr])

        # default reputation/energy/data_size
        reps = np.array([meta.get('reputation', 1.0) for meta in metas], dtype=float)
        energies = np.array([meta.get('energy', 1.0) for meta in metas], dtype=float)
        data_sizes = np.array([meta.get('data_size', 1.0) for meta in metas], dtype=float)

        # system weight: prefer lower energy cost and larger data_size
        # produce normalized energy weight (inverse)
        energy_weight = 1.0 / (energies + 1e-9)
        data_weight = data_sizes / (np.sum(data_sizes) + 1e-9)

        # combine into final weight
        # base weight: similarity * reputation
        base = sims * reps
        # combine with system weights (alpha balances system vs stat)
        sys_term = (energy_weight * data_weight)
        weights = (1.0 - self.alpha) * base + self.alpha * sys_term

        # clamp negative weights to small positive to avoid elimination
        weights = np.clip(weights, 1e-6, None)
        weights = weights / (weights.sum() + 1e-9)

        # weighted sum
        agg_vec = np.sum(arr * weights[:, None], axis=0)

        # convert back
        if original_format == 'state_dict' and state_ref is not None:
            return self._unflatten_to_state_dict(agg_vec, state_ref)
        elif original_format == 'tensor':
            return torch.from_numpy(agg_vec)
        else:
            return agg_vec

    def _flatten_state_dict(self, state_dict):
        """Flatten a state dict to a 1D numpy array."""
        vecs = []
        for key in sorted(state_dict.keys()):
            v = state_dict[key]
            if isinstance(v, torch.Tensor):
                vecs.append(v.detach().cpu().numpy().reshape(-1))
            else:
                vecs.append(np.array(v).reshape(-1))
        return np.concatenate(vecs)

    def _unflatten_to_state_dict(self, vector, reference_state_dict):
        """Unflatten a 1D vector back to state dict format."""
        result = {}
        idx = 0
        for key in sorted(reference_state_dict.keys()):
            ref = reference_state_dict[key]
            if isinstance(ref, torch.Tensor):
                size = ref.numel()
                shape = tuple(ref.shape)
            else:
                tmp = np.array(ref)
                size = tmp.size
                shape = tmp.shape
            slice_ = vector[idx:idx + size]
            param = slice_.reshape(shape)
            if isinstance(ref, torch.Tensor):
                result[key] = torch.from_numpy(param).to(dtype=ref.dtype)
            else:
                result[key] = param
            idx += size
        return result
