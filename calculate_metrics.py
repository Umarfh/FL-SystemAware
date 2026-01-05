import pandas as pd
import numpy as np
import json

# =========================
# Load files
# =========================
# Load training_metrics.json which now contains all necessary data
with open("training_metrics.json", "r") as f:
    training_data = json.load(f)

df = pd.DataFrame(training_data["rounds"])

# Ensure numeric types for relevant columns
for col in ["round", "test_acc", "test_loss", "anomaly_score", "rejection_ratio", "communication_cost_kb", "energy_consumption_mj"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# For reputation score, we can still derive it as a proxy if not directly logged
WINDOW = 5
EPS = 1e-12
df["reputation_score"] = df["test_acc"].rolling(WINDOW, min_periods=1).mean()


# =========================
# Joint efficiency–security metrics
# =========================

# 1. Communication efficiency under anomaly
# Use the newly calculated communication_cost_kb
df["secure_comm_efficiency"] = (
    df["test_acc"] / (df["communication_cost_kb"] + EPS) / (1.0 + df["anomaly_score"] + EPS)
)

# 2. Energy efficiency under anomaly
# Use the newly calculated energy_consumption_mj
df["secure_energy_efficiency"] = (
    df["test_acc"] / (df["energy_consumption_mj"] + EPS) / (1.0 + df["anomaly_score"] + EPS)
)

# 3. Reputation-weighted efficiencies
df["rep_weighted_comm_eff"] = (
    df["test_acc"] / (df["communication_cost_kb"] + EPS) * df["reputation_score"]
)

df["rep_weighted_energy_eff"] = (
    df["test_acc"] / (df["energy_consumption_mj"] + EPS) * df["reputation_score"]
)

# 4. Defense-adjusted efficiency (accounts for rejection)
df["defense_adjusted_comm_eff"] = (
    df["test_acc"] / (df["communication_cost_kb"] + EPS) * (1.0 - df["rejection_ratio"])
)

df["defense_adjusted_energy_eff"] = (
    df["test_acc"] / (df["energy_consumption_mj"] + EPS) * (1.0 - df["rejection_ratio"])
)

# =========================
# Save unified metrics
# =========================
# Save to metrics_with_anomaly_reputation.csv as it now contains all relevant metrics
df.to_csv("metrics_with_anomaly_reputation.csv", index=False)

print("DONE: metrics_with_anomaly_reputation.csv created with dynamic efficiency metrics.")
