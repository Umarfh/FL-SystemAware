"""
Defense Pipeline Module for Federated Learning
Coordinates detection, filtering, and reputation management to defend against malicious updates
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from defenses.detector import Detector
from defenses.filter import Filter
from defenses.reputation import ReputationManager


@dataclass
class DefenseMetrics:
    """Metrics tracking defense performance"""
    total_updates: int = 0
    rejected_updates: int = 0
    clipped_updates: int = 0
    accepted_updates: int = 0
    avg_anomaly_score: float = 0.0
    max_anomaly_score: float = 0.0
    min_anomaly_score: float = 1.0
    detection_rate: float = 0.0
    round_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def update(self, rejected: int, clipped: int, accepted: int, scores: List[float]):
        """Update metrics for current round"""
        total = rejected + clipped + accepted
        self.total_updates += total
        self.rejected_updates += rejected
        self.clipped_updates += clipped
        self.accepted_updates += accepted
        
        if scores:
            self.avg_anomaly_score = np.mean(scores)
            self.max_anomaly_score = max(scores)
            self.min_anomaly_score = min(scores)
        
        self.detection_rate = self.rejected_updates / max(1, self.total_updates)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'total_updates': self.total_updates,
            'rejected_updates': self.rejected_updates,
            'clipped_updates': self.clipped_updates,
            'accepted_updates': self.accepted_updates,
            'rejection_rate': self.detection_rate,
            'avg_anomaly_score': self.avg_anomaly_score,
            'max_anomaly_score': self.max_anomaly_score
        }


class DefensePipeline:
    """
    Coordinated defense pipeline for federated learning.
    
    Pipeline stages:
    1. Detection: Identify anomalous updates using various detectors
    2. Reputation: Track client trustworthiness over time
    3. Filtering: Sanitize, clip, or reject updates based on scores
    4. Aggregation: Combine cleaned updates with trust-weighted averaging
    
    Features:
    - Multi-stage defense with configurable modules
    - Comprehensive logging and metrics tracking
    - Adaptive thresholds based on round history
    - Client reputation management
    - Support for multiple detection methods
    - Graceful degradation when attacks detected
    
    Args:
        args: Configuration object containing defense parameters
        logger: Optional logger for detailed output
    """
    
    def __init__(self, args, logger: Optional[logging.Logger] = None):
        self.args = args
        self.logger = logger or self._setup_default_logger()
        
        # Parse defense parameters
        self.defense_params = self._parse_defense_params(args)
        
        # Initialize defense modules
        self._init_detector()
        self._init_filter()
        self._init_reputation()
        
        # Metrics and monitoring
        self.metrics = DefenseMetrics()
        self.current_round = 0
        self.adaptive_threshold = self.defense_params.get("adaptive_threshold", True)
        
        # Client tracking
        self.client_history = defaultdict(list)
        self.suspicious_clients = set()
        
        # Configuration
        self.min_accepted_updates = self.defense_params.get("min_accepted_updates", 1)
        self.enable_logging = self.defense_params.get("enable_logging", True)
        self.verbose = self.defense_params.get("verbose", False)
        
        self.logger.info("Defense pipeline initialized successfully")
        self._log_configuration()
    
    def sanitize_updates(self, updates: List[np.ndarray], 
                        client_ids: Optional[List[int]] = None,
                        global_model: Optional[np.ndarray] = None) -> List[Optional[np.ndarray]]:
        """
        Sanitize client updates through the defense pipeline.
        
        Args:
            updates: List of numpy arrays representing client updates
            client_ids: Optional list of client IDs (defaults to indices)
            global_model: Optional global model for context-aware detection
            
        Returns:
            List of sanitized updates (None for rejected updates)
        """
        # Convert updates to a list if it's a numpy array with more than one element
        if isinstance(updates, np.ndarray) and updates.ndim > 1:
            updates = list(updates)

        if not updates:
            self.logger.warning("No updates provided to sanitize")
            return []
        
        # Set default client IDs
        if client_ids is None:
            client_ids = list(range(len(updates)))
        
        if len(updates) != len(client_ids):
            raise ValueError(f"Mismatch: {len(updates)} updates vs {len(client_ids)} client_ids")
        
        self.current_round += 1
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Defense Pipeline - Round {self.current_round}")
        self.logger.info(f"Processing {len(updates)} updates")
        self.logger.info(f"{'='*60}")
        
        # Stage 1: Detection
        anomaly_scores = self._detect_anomalies(updates, client_ids, global_model)
        
        # Stage 2: Reputation
        reputations = self._get_reputations(client_ids)
        
        # Stage 3: Filtering
        sanitized_updates, filter_stats = self._filter_updates(
            updates, client_ids, anomaly_scores, reputations
        )
        
        # Stage 4: Update reputation
        self._update_reputations(client_ids, anomaly_scores)
        
        # Track metrics
        self._update_metrics(filter_stats, anomaly_scores)
        
        # Log round summary
        self._log_round_summary(filter_stats, anomaly_scores, reputations)
        
        # Adaptive threshold adjustment
        if self.adaptive_threshold:
            self._adjust_thresholds(filter_stats)
        
        return sanitized_updates
    
    def aggregate_with_trust(self, sanitized_updates: List[Optional[np.ndarray]],
                           client_ids: List[int],
                           trust_weighting: bool = True) -> Optional[np.ndarray]:
        """
        Aggregate sanitized updates with trust-based weighting.
        
        Args:
            sanitized_updates: List of sanitized updates (None for rejected)
            client_ids: List of client IDs
            trust_weighting: Whether to weight by reputation
            
        Returns:
            Aggregated update vector or None if no valid updates
        """
        valid_updates = []
        weights = []
        
        for update, cid in zip(sanitized_updates, client_ids):
            if update is not None:
                valid_updates.append(update)
                if trust_weighting:
                    weights.append(self.reputation.get(cid))
                else:
                    weights.append(1.0)
        
        if not valid_updates:
            self.logger.error("No valid updates to aggregate!")
            return None
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / (weights.sum() + 1e-9)
        
        # Weighted average
        aggregated = np.sum([w * u for w, u in zip(weights, valid_updates)], axis=0)
        
        self.logger.info(f"Aggregated {len(valid_updates)}/{len(sanitized_updates)} updates")
        if trust_weighting:
            self.logger.info(f"Trust weights range: [{weights.min():.3f}, {weights.max():.3f}]")
        
        return aggregated
    
    # ==================== Detection Stage ====================
    
    def _detect_anomalies(self, updates: List[np.ndarray], 
                         client_ids: List[int],
                         global_model: Optional[np.ndarray]) -> List[float]:
        """
        Detect anomalies in client updates.
        
        Returns:
            List of anomaly scores [0, 1] for each update
        """
        self.logger.info("\n[Stage 1: Detection]")
        
        anomaly_scores = []
        for i, (update, cid) in enumerate(zip(updates, client_ids)):
            # Get anomaly score from detector
            score = self.detector.score(
                update, 
                client_id=cid,
                global_model=global_model,
                client_history=self.client_history.get(cid, [])
            )
            
            anomaly_scores.append(score)
            
            if self.verbose:
                self.logger.debug(f"  Client {cid}: anomaly score = {score:.4f}")
            
            # Track in client history
            self.client_history[cid].append({
                'round': self.current_round,
                'anomaly_score': score,
                'update_norm': np.linalg.norm(update)
            })
        
        avg_score = np.mean(anomaly_scores)
        max_score = np.max(anomaly_scores)
        self.logger.info(f"  Anomaly scores: avg={avg_score:.4f}, max={max_score:.4f}")
        
        return anomaly_scores
    
    def _get_reputations(self, client_ids: List[int]) -> List[float]:
        """Get current reputation scores for clients"""
        reputations = [self.reputation.get(cid) for cid in client_ids]
        
        avg_rep = np.mean(reputations)
        min_rep = np.min(reputations)
        self.logger.info(f"\n[Stage 2: Reputation]")
        self.logger.info(f"  Avg reputation: {avg_rep:.3f}, Min: {min_rep:.3f}")
        
        return reputations
    
    def _filter_updates(self, updates: List[np.ndarray],
                       client_ids: List[int],
                       anomaly_scores: List[float],
                       reputations: List[float]) -> Tuple[List[Optional[np.ndarray]], Dict[str, int]]:
        """
        Filter updates based on anomaly scores and reputations.
        
        Returns:
            Tuple of (sanitized_updates, statistics_dict)
        """
        self.logger.info(f"\n[Stage 3: Filtering]")
        
        sanitized = []
        stats = {'rejected': 0, 'clipped': 0, 'accepted': 0}
        
        for update, cid, score, rep in zip(updates, client_ids, anomaly_scores, reputations):
            result = self.filter.sanitize(
                update,
                anomaly_score=score,
                reputation=rep,
                client_id=cid
            )
            
            if result is None:
                sanitized.append(None)
                stats['rejected'] += 1
                self.suspicious_clients.add(cid)
                if self.verbose:
                    self.logger.debug(f"  Client {cid}: REJECTED (score={score:.3f}, rep={rep:.3f})")
            elif not np.array_equal(result, update):
                sanitized.append(result)
                stats['clipped'] += 1
                if self.verbose:
                    self.logger.debug(f"  Client {cid}: CLIPPED (score={score:.3f}, rep={rep:.3f})")
            else:
                sanitized.append(result)
                stats['accepted'] += 1
                if self.verbose:
                    self.logger.debug(f"  Client {cid}: ACCEPTED (score={score:.3f}, rep={rep:.3f})")
        
        self.logger.info(f"  Results: {stats['accepted']} accepted, "
                        f"{stats['clipped']} clipped, {stats['rejected']} rejected")
        
        # Safety check: ensure minimum accepted updates
        if stats['accepted'] + stats['clipped'] < self.min_accepted_updates:
            self.logger.warning(f"  WARNING: Only {stats['accepted'] + stats['clipped']} valid updates "
                              f"(minimum: {self.min_accepted_updates}). Loosening filter thresholds.")
            self.filter.loosen_thresholds()
            # Re-filter with loosened thresholds
            sanitized = []
            stats = {'rejected': 0, 'clipped': 0, 'accepted': 0}
            for update, cid, score, rep in zip(updates, client_ids, anomaly_scores, reputations):
                result = self.filter.sanitize(
                    update,
                    anomaly_score=score,
                    reputation=rep,
                    client_id=cid
                )
                if result is None:
                    sanitized.append(None)
                    stats['rejected'] += 1
                    self.suspicious_clients.add(cid)
                elif not np.array_equal(result, update):
                    sanitized.append(result)
                    stats['clipped'] += 1
                else:
                    sanitized.append(result)
                    stats['accepted'] += 1
            self.logger.info(f"  Re-filtered results: {stats['accepted']} accepted, "
                            f"{stats['clipped']} clipped, {stats['rejected']} rejected")
        
        return sanitized, stats
    
    def _update_reputations(self, client_ids: List[int], 
                           anomaly_scores: List[float]) -> None:
        """Update reputation scores based on anomaly scores"""
        for cid, score in zip(client_ids, anomaly_scores):
            self.reputation.update(cid, score)
    
    # ==================== Metrics & Monitoring ====================
    
    def _update_metrics(self, filter_stats: Dict[str, int], 
                       anomaly_scores: List[float]) -> None:
        """Update defense metrics"""
        self.metrics.update(
            rejected=filter_stats['rejected'],
            clipped=filter_stats['clipped'],
            accepted=filter_stats['accepted'],
            scores=anomaly_scores
        )
        
        # Store round history
        self.metrics.round_history.append({
            'round': self.current_round,
            'rejected': filter_stats['rejected'],
            'clipped': filter_stats['clipped'],
            'accepted': filter_stats['accepted'],
            'avg_anomaly_score': np.mean(anomaly_scores),
            'max_anomaly_score': np.max(anomaly_scores)
        })
    
    def _log_round_summary(self, filter_stats: Dict[str, int],
                          anomaly_scores: List[float],
                          reputations: List[float]) -> None:
        """Log summary of defense round"""
        if not self.enable_logging:
            return
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Round {self.current_round} Summary")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Filter Results:")
        self.logger.info(f"  Accepted: {filter_stats['accepted']}")
        self.logger.info(f"  Clipped:  {filter_stats['clipped']}")
        self.logger.info(f"  Rejected: {filter_stats['rejected']}")
        self.logger.info(f"Anomaly Scores:")
        self.logger.info(f"  Mean:   {np.mean(anomaly_scores):.4f}")
        self.logger.info(f"  Median: {np.median(anomaly_scores):.4f}")
        self.logger.info(f"  Max:    {np.max(anomaly_scores):.4f}")
        self.logger.info(f"Reputations:")
        self.logger.info(f"  Mean:   {np.mean(reputations):.3f}")
        self.logger.info(f"  Min:    {np.min(reputations):.3f}")
        
        if self.suspicious_clients:
            self.logger.info(f"Suspicious Clients: {sorted(self.suspicious_clients)}")
    
    def _adjust_thresholds(self, filter_stats: Dict[str, int]) -> None:
        """Adaptively adjust filter thresholds based on rejection rate"""
        rejection_rate = filter_stats['rejected'] / max(1, sum(filter_stats.values()))
        
        if rejection_rate > 0.5:
            # Too many rejections, relax thresholds
            self.filter.loosen_thresholds()
            self.logger.info("  [Adaptive] Loosening filter thresholds (high rejection rate)")
        elif rejection_rate < 0.1 and self.current_round > 10:
            # Very few rejections, tighten thresholds
            self.filter.tighten_thresholds()
            self.logger.info("  [Adaptive] Tightening filter thresholds (low rejection rate)")
    
    # ==================== Initialization & Configuration ====================
    
    def _parse_defense_params(self, args) -> Dict[str, Any]:
        """Parse defense parameters from args"""
        dp = getattr(args, "defense_params", None)
        
        if dp is None:
            dp = {}
        elif isinstance(dp, str):
            try:
                import ast
                dp = ast.literal_eval(dp)
            except Exception as e:
                self.logger.warning(f"Failed to parse defense_params: {e}")
                dp = {}
        
        # Set defaults
        defaults = {
            'detector': 'rep_pca',
            'detector_params': {},
            'filter_params': {},
            'rep_init': 1.0,
            'rep_decay': 0.95,
            'rep_incr': 0.01,
            'rep_decr': 0.1,
            'adaptive_threshold': True,
            'min_accepted_updates': 1,
            'enable_logging': True,
            'verbose': False
        }
        
        # Merge with defaults
        for key, value in defaults.items():
            if key not in dp:
                dp[key] = value
        
        return dp
    
    def _init_detector(self) -> None:
        """Initialize anomaly detector"""
        detector_method = self.defense_params['detector']
        detector_params = self.defense_params['detector_params']
        
        self.detector = Detector(
            method=detector_method,
            params=detector_params
        )
        
        self.logger.info(f"Initialized detector: {detector_method}")
    
    def _init_filter(self) -> None:
        """Initialize update filter"""
        filter_params = self.defense_params['filter_params']
        
        self.filter = Filter(params=filter_params)
        
        self.logger.info("Initialized filter")
    
    def _init_reputation(self) -> None:
        """Initialize reputation manager"""
        self.reputation = ReputationManager(
            init=self.defense_params['rep_init'],
            decay=self.defense_params['rep_decay'],
            incr=self.defense_params['rep_incr'],
            decr=self.defense_params['rep_decr']
        )
        
        self.logger.info("Initialized reputation manager")
    
    def _setup_default_logger(self) -> logging.Logger:
        """Setup default logger"""
        logger = logging.getLogger("DefensePipeline")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(levelname)s] %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _log_configuration(self) -> None:
        """Log defense configuration"""
        self.logger.info("\nDefense Configuration:")
        self.logger.info(f"  Detector: {self.defense_params['detector']}")
        self.logger.info(f"  Reputation init: {self.defense_params['rep_init']:.2f}")
        self.logger.info(f"  Reputation decay: {self.defense_params['rep_decay']:.2f}")
        self.logger.info(f"  Adaptive thresholds: {self.adaptive_threshold}")
        self.logger.info(f"  Min accepted updates: {self.min_accepted_updates}")
    
    # ==================== Public API ====================
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get defense metrics"""
        return self.metrics.to_dict()
    
    def get_client_reputation(self, client_id: int) -> float:
        """Get reputation for specific client"""
        return self.reputation.get(client_id)
    
    def get_suspicious_clients(self) -> List[int]:
        """Get list of suspicious client IDs"""
        return sorted(self.suspicious_clients)
    
    def reset_client_reputation(self, client_id: int) -> None:
        """Reset reputation for a specific client"""
        self.reputation.reset(client_id)
        self.client_history[client_id] = []
        self.suspicious_clients.discard(client_id)
        self.logger.info(f"Reset reputation for client {client_id}")
    
    def reset_all(self) -> None:
        """Reset all defense state"""
        self.reputation.reset()
        self.client_history.clear()
        self.suspicious_clients.clear()
        self.metrics = DefenseMetrics()
        self.current_round = 0
        self.logger.info("Defense pipeline reset")
    
    def __repr__(self) -> str:
        return (f"DefensePipeline(detector={self.defense_params['detector']}, "
                f"rounds={self.current_round}, "
                f"rejection_rate={self.metrics.detection_rate:.2%})")
