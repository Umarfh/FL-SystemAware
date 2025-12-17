import numpy as np # Added import for numpy
from collections import defaultdict
from sklearn.decomposition import PCA # Assuming PCA is used and needs to be imported

class Detector:
    """Anomaly detector for client updates."""
    
    def __init__(self, method="rep_pca", params=None):
        self.method = method
        self.params = params or {}
        self.history = defaultdict(list)
        self.pca = None
        self.threshold = self.params.get("detector_threshold", 2.0)
        
    def score(self, update, client_id=None, global_model=None, client_history=None):
        """
        Compute anomaly score for an update.
        Returns: float in [0, 1] where higher = more anomalous
        """
        if self.method == "rep_pca":
            return self._rep_pca_score(update, client_id, global_model, client_history)
        elif self.method == "norm":
            return self._norm_score(update, global_model, client_history)
        else:
            return 0.0  # No detection
    
    def _norm_score(self, update, global_model, client_history):
        """Simple norm-based scoring."""
        norm = np.linalg.norm(update)
        # Store history for comparison
        if not hasattr(self, '_norm_history'):
            self._norm_history = []
        self._norm_history.append(norm)
        
        if len(self._norm_history) < 2:
            return 0.0
        
        median = np.median(self._norm_history[-20:])  # Last 20 updates
        mad = np.median(np.abs(np.array(self._norm_history[-20:]) - median)) + 1e-9
        z_score = abs(norm - median) / mad
        
        # Convert to [0, 1] with sigmoid
        return 1.0 / (1.0 + np.exp(-0.5 * (z_score - self.threshold)))
    
    def _rep_pca_score(self, update, client_id, global_model, client_history):
        """Reputation-weighted PCA-based detection."""
        # Store update in history
        if client_id is not None:
            self.history[client_id].append(update)
        
        # Need multiple updates to do PCA
        all_updates = []
        for updates in self.history.values():
            all_updates.extend(updates[-10:])  # Last 10 per client
        
        if len(all_updates) < 5:
            return 0.0  # Not enough data
        
        try:
            # Fit PCA if needed
            if self.pca is None or len(all_updates) % 10 == 0:
                n_components = min(3, len(all_updates) - 1)
                self.pca = PCA(n_components=n_components)
                self.pca.fit(all_updates)
            
            # Reconstruction error as anomaly score
            reconstructed = self.pca.inverse_transform(self.pca.transform([update]))
            error = np.linalg.norm(update - reconstructed[0])
            
            # Normalize by typical reconstruction error
            if not hasattr(self, '_error_history'):
                self._error_history = []
            self._error_history.append(error)
            
            if len(self._error_history) < 2:
                return 0.0
            
            median_error = np.median(self._error_history[-20:])
            mad_error = np.median(np.abs(np.array(self._error_history[-20:]) - median_error)) + 1e-9
            z_score = (error - median_error) / mad_error
            
            # Convert to [0, 1]
            return 1.0 / (1.0 + np.exp(-0.5 * (z_score - self.threshold)))
        
        except Exception as e:
            print(f"[Detector] PCA failed: {e}")
            return 0.0
