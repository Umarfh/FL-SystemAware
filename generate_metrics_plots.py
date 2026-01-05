import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from plot_utils import parse_logs, plot_accuracy, plot_loss, plot_asr, plot_round_time, \
                       plot_communication_cost, plot_energy_consumption, \
                       plot_anomaly_and_reputation_scores_vs_round, plot_rejection_ratio_vs_round, \
                       plot_active_clients, plot_rejected_accepted_updates

# =========================
# Load metrics
# =========================
log_file = "training_metrics.json" # Assuming this is in the current directory
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

epochs, accs, losses, asrs, asr_losses, round_times, active_clients, stragglers, dropouts, \
communication_costs, energy_consumptions, anomaly_scores_per_round, \
reputation_scores, rejection_ratios_per_round = parse_logs(log_file)

# =========================
# PLOTS
# =========================
print("Generating plots...")

plot_accuracy(epochs, accs, output_dir)
plot_loss(epochs, losses, output_dir)
plot_asr(epochs, asrs, output_dir)
plot_round_time(epochs, round_times, output_dir)
plot_rejected_accepted_updates(epochs, active_clients, dropouts, output_dir)
plot_communication_cost(epochs, communication_costs, output_dir)
plot_energy_consumption(epochs, energy_consumptions, output_dir)

# Plot new defense metrics
plot_anomaly_and_reputation_scores_vs_round(epochs, anomaly_scores_per_round, reputation_scores, output_dir)
plot_rejection_ratio_vs_round(epochs, rejection_ratios_per_round, output_dir)
plot_active_clients(epochs, active_clients, output_dir)

print("DONE: Figures generated in the 'plots' directory.")
