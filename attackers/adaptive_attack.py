# attackers/adaptive_attack.py
"""
AdaptiveAttack implementation for FLPoison-main.
Provides simple adaptive multi-round attacker behavior
and supports both omniscient and non-omniscient interfaces expected by the framework.
"""
import numpy as np
import logging
from attackers import attacker_registry
from global_utils import actor
from fl.client import Client

logger = logging.getLogger(__name__)

@attacker_registry.register("AdaptiveAttack", ["model_poisoning", "non_omniscient"], "attacker")
@actor('attacker', 'model_poisoning', 'non_omniscient')
class AdaptiveAttack(Client):
    """
    A compact adaptive attack example:
    - omniscient(): used by orchestrator/server to request crafted updates
    - non_omniscient(): modify local client.update before send (scaling)
    - adaptive_round(): update strategy across rounds (very simple)
    """
    def __init__(self, *args, attack_alpha=1.0, scaling_factor=10.0, **kwargs):
        super().__init__(*args, **kwargs)
        # attack hyper-params
        self.attack_alpha = float(getattr(self.args, "attack_alpha", attack_alpha))
        self.scaling_factor = float(getattr(self.args, "attack_scaling", scaling_factor))
        # simple internal state
        self.strategy = "benign"  # will flip to 'sign_flip' when adaptive
        self.round_seen = -1

    def adaptive_round(self, round_num, server=None, clients=None):
        """
        Called by orchestrator to let attackers adapt strategy over rounds.
        Example heuristic: switch to sign-flip after some rounds or condition.
        """
        self.round_seen = round_num
        # simple rule: flip if attack_alpha > 0.9 and round > 10
        if self.attack_alpha >= 1.0 and round_num >= getattr(self.args, "attack_start_epoch", 10):
            self.strategy = "sign_flip"
        else:
            self.strategy = "scale"

    def omniscient(self, active_clients=None):
        """
        Generate malicious update(s) with omniscient knowledge (server calls this).
        Returns a weight vector (numpy) shaped like model vector.
        For simplicity, craft a scaled negative of global update (directional).
        """
        try:
            global_vec = self.global_weights_vec
            if global_vec is None:
                return None
            # create a simple poisoned direction (sign flip or scaling)
            if self.strategy == "sign_flip":
                poisoned = -1.0 * global_vec * self.scaling_factor
            else:
                poisoned = global_vec + self.scaling_factor * (global_vec - global_vec.mean())
            return poisoned
        except Exception as e:
            logger.exception("AdaptiveAttack.omniscient failed: %s", e)
            return None

    def non_omniscient(self):
        """
        Local post-processing before client.submit:
        - scale update by scaling_factor or flip sign depending on strategy.
        """
        try:
            u = self.update
            if u is None:
                return None
            # if torch tensor -> convert to numpy then back (preserve type)
            is_torch = hasattr(u, "cpu") and not isinstance(u, np.ndarray)
            if is_torch:
                import torch
                u_np = u.cpu().numpy()
            else:
                u_np = np.array(u, copy=True)

            if self.strategy == "sign_flip":
                out = -1.0 * u_np * self.scaling_factor
            else:
                out = u_np * (1.0 + 0.1 * self.scaling_factor)

            if is_torch:
                return torch.from_numpy(out).to(u.dtype)
            return out
        except Exception as e:
            logger.exception("AdaptiveAttack.non_omniscient failed: %s", e)
            return self.update
