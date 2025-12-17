# defenses/filter.py
import numpy as np
import logging

class Filter:
    """Filter/sanitize updates based on anomaly scores and reputation."""
    
    def __init__(self, params=None):
        self.params = params or {}
        self.rejection_threshold = self.params.get("rejection_threshold", 0.8)
        self.clipping_threshold = self.params.get("clipping_threshold", 0.5)
        self.reputation_threshold = self.params.get("reputation_threshold", 0.3) # Added reputation threshold
        self.clip_norm = self.params.get("clip_norm", 10.0)
        
        # Parameters for adaptive control
        self.threshold_step = self.params.get("threshold_step", 0.05) # How much to adjust thresholds by
        self.min_rejection_threshold = self.params.get("min_rejection_threshold", 0.5)
        self.max_rejection_threshold = self.params.get("max_rejection_threshold", 0.95)
        self.min_reputation_threshold = self.params.get("min_reputation_threshold", 0.1)
        self.max_reputation_threshold = self.params.get("max_reputation_threshold", 0.5)
        self.logger = logging.getLogger("Filter")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def tighten_thresholds(self):
        """Tighten filtering thresholds, making the filter stricter."""
        self.rejection_threshold = min(self.max_rejection_threshold, self.rejection_threshold + self.threshold_step)
        self.reputation_threshold = min(self.max_reputation_threshold, self.reputation_threshold + self.threshold_step)
        self.logger.info(f"[Filter] Tightened thresholds: rejection={self.rejection_threshold:.3f}, reputation={self.reputation_threshold:.3f}")

    def loosen_thresholds(self):
        """Loosen filtering thresholds, making the filter more permissive."""
        self.rejection_threshold = max(self.min_rejection_threshold, self.rejection_threshold - self.threshold_step)
        self.reputation_threshold = max(self.min_reputation_threshold, self.reputation_threshold - self.threshold_step)
        self.logger.info(f"[Filter] Loosened thresholds: rejection={self.rejection_threshold:.3f}, reputation={self.reputation_threshold:.3f}")

    def sanitize(self, update, anomaly_score, reputation, client_id=None):
        """
        Sanitize an update based on anomaly score and reputation.
        
        Returns:
            - None if update should be rejected
            - Modified update otherwise
        """
        # Reject very anomalous updates
        if anomaly_score > self.rejection_threshold:
            self.logger.info(f"[Filter] Rejecting update: anomaly={anomaly_score:.3f} (threshold={self.rejection_threshold:.3f})")
            return None
        
        # Reject updates from low-reputation clients
        if reputation < self.reputation_threshold:
            self.logger.info(f"[Filter] Rejecting update: reputation={reputation:.3f} (threshold={self.reputation_threshold:.3f})")
            return None
        
        # Clip moderately anomalous updates
        if anomaly_score > self.clipping_threshold:
            norm = np.linalg.norm(update)
            if norm > self.clip_norm:
                clipped = update * (self.clip_norm / norm)
                self.logger.info(f"[Filter] Clipping update: norm={norm:.3f} -> {self.clip_norm:.3f}")
                return clipped
        
        return update
