# defenses/reputation.py
class ReputationManager:
    def __init__(self, init=1.0, decay=0.95, incr=0.01, decr=0.1, 
                 min_score=0.0, max_score=1.0, anomaly_threshold_decr=0.6): # Added anomaly_threshold_decr
        self.scores = {}  # client_id -> float
        self.init = init
        self.decay = decay
        self.incr = incr
        self.decr = decr
        self.min_score = min_score
        self.max_score = max_score
        self.anomaly_threshold_decr = anomaly_threshold_decr # Threshold above which reputation decreases

    def ensure(self, client_id):
        if client_id not in self.scores:
            self.scores[client_id] = self.init

    def update(self, client_id, anomaly_score):
        self.ensure(client_id)
        # anomaly_score in [0,1], high means suspicious
        if anomaly_score > self.anomaly_threshold_decr:
            # More aggressive decrease for higher anomaly scores
            decrease_amount = self.decr * (anomaly_score - self.anomaly_threshold_decr) / (1 - self.anomaly_threshold_decr)
            self.scores[client_id] = max(self.min_score, self.scores[client_id] - decrease_amount)
        else:
            # Gradual increase and decay for low anomaly scores
            self.scores[client_id] = min(self.max_score, self.scores[client_id]*self.decay + self.incr*(1-anomaly_score))

    def get(self, client_id):
        self.ensure(client_id)
        return self.scores[client_id]
