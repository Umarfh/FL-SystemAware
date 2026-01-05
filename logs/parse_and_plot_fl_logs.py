import re
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sys
import os
import json

# Add the parent directory to sys.path to import plot_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_utils import setup_paper_style

LOG_PATH = "logs/FedOpt/MNIST_lenet/iid/MNIST_lenet_iid_NoAttack_SystemAware_50_50_0.005_FedOpt/training_metrics.json"
PLOTS_DIR = "plots/" # Define a dedicated directory for plots
CSV_PATH = Path(PLOTS_DIR) / "metrics.csv" # Save CSV in the plots directory

def parse_log(path):
    metrics = []
    path_obj = Path(path)

    if path_obj.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "rounds" in data: # Handle experiment_metrics.json structure
                for round_data in data["rounds"]:
                    current = {"round": round_data["round"] + 1} # Adjust round to be 1-indexed
                    current["train_acc"] = round_data.get("train_acc")
                    current["train_loss"] = round_data.get("train_loss")
                    current["val_acc"] = round_data.get("test_acc") * 100 if round_data.get("test_acc") is not None else None # Convert to percentage
                    current["val_loss"] = round_data.get("test_loss")
                    current["comm_kb"] = round_data.get("communication_cost_kb")
                    current["energy_mj"] = round_data.get("energy_consumption_mj")
                    current["anomaly_score"] = round_data.get("anomaly_score")
                    current["rejection_ratio"] = round_data.get("rejection_ratio")
                    current["reputation_score"] = round_data.get("reputation_score") # Assuming this is now logged
                    current["active_clients"] = round_data.get("active_clients")
                    metrics.append(current)
            else: # Handle training_metrics.json structure (top-level arrays) - this structure is likely deprecated
                comm_kb_per_round = data.get("comm_kb_per_round", [])
                energy_per_round = data.get("energy_per_round", [])
                test_accuracies = data.get("test_accuracies", [])
                test_losses = data.get("test_losses", [])
                train_accuracies = data.get("train_accuracies", [])
                train_losses = data.get("train_losses", [])

                num_rounds = max(len(comm_kb_per_round), len(energy_per_round), len(test_accuracies))

                for i in range(num_rounds):
                    current = {"round": i + 1}
                    if i < len(comm_kb_per_round):
                        current["comm_kb"] = comm_kb_per_round[i]
                    if i < len(energy_per_round):
                        current["energy_mj"] = energy_per_round[i]
                    if i < len(test_accuracies):
                        current["val_acc"] = test_accuracies[i] * 100 # Convert to percentage
                    if i < len(test_losses):
                        current["val_loss"] = test_losses[i]
                    if i < len(train_accuracies):
                        current["train_acc"] = train_accuracies[i]
                    if i < len(train_losses):
                        current["train_loss"] = train_losses[i]
                    metrics.append(current)
    else: # Assume it's a .txt log file
        current = {}
        # No longer expecting anomaly_mean or reputation_mean from .txt logs as they are in JSON
        # The .txt log parsing is less robust and should ideally be replaced by JSON parsing.

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                # --- New round (Epoch block) ---
                m_epoch = re.search(r"Round\s+(\d+)\s*Test Acc:\s*([\d.]+)\s*Test Loss:\s*([\d.]+)\s*Comm \(KB\):\s*([\d.]+)\s*Energy \(mJ\):\s*([\d.]+)\s*Anomaly Score:\s*([\d.]+)\s*Rejection Ratio:\s*([\d.]+)", line)
                if m_epoch:
                    if "round" in current:
                        metrics.append(current)
                        current = {}
                    current["round"] = int(m_epoch.group(1))
                    current["val_acc"] = float(m_epoch.group(2)) * 100 # Convert to percentage
                    current["val_loss"] = float(m_epoch.group(3))
                    current["comm_kb"] = float(m_epoch.group(4))
                    current["energy_mj"] = float(m_epoch.group(5))
                    current["anomaly_score"] = float(m_epoch.group(6))
                    current["rejection_ratio"] = float(m_epoch.group(7))
                    # Active clients and reputation score are not easily parsed from this single line
                    # If needed, more complex regex or multi-line parsing would be required.
                    continue
                
                # Fallback for older log formats if needed, but prioritize the new structured log line
                m_old_epoch = re.search(r"Epoch\s+(\d+)\s*Train Acc:\s*([\d.]+)\s*Train loss:\s*([\d.]+)\s*Test Acc:\s*([\d.]+)\s*Test loss:\s*([\d.]+)", line)
                if m_old_epoch:
                    if "round" in current:
                        metrics.append(current)
                        current = {}
                    current["round"] = int(m_old_epoch.group(1))
                    current["train_acc"] = float(m_old_epoch.group(2))
                    current["train_loss"] = float(m_old_epoch.group(3))
                    current["val_acc"] = float(m_old_old_epoch.group(4)) * 100 # Convert to percentage
                    current["val_loss"] = float(m_old_epoch.group(5))
                    continue

                m_comm = re.search(r"Communication \(KB\):\s*([\d.]+)", line)
                if m_comm:
                    current["comm_kb"] = float(m_comm.group(1))
                    continue

                m_energy = re.search(r"Energy \(mJ\):\s*([\d.]+)", line)
                if m_energy:
                    current["energy_mj"] = float(m_energy.group(1))
                    continue

                m_rejected = re.search(r"Rejected ratio:\s*([\d.]+)", line)
                if m_rejected:
                    current["rejection_ratio"] = float(m_rejected.group(1))
                    continue

        # store last round
        if "round" in current:
            metrics.append(current)

    return metrics


def save_csv(metrics, path):
    if not metrics:
        print("No metrics parsed.")
        return

    # Collect all keys that appear
    keys = set()
    for m in metrics:
        keys.update(m.keys())
    # Ensure round is first
    keys = ["round"] + sorted(k for k in keys if k != "round")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for m in metrics:
            writer.writerow(m)

    print(f"Saved metrics to {path}")


def get_xy(metrics, key):
    xs, ys = [], []
    for m in metrics:
        if key in m and m[key] is not None:
            xs.append(m["round"])
            ys.append(m[key])
    return xs, ys

def plot_cumulative_metrics(metrics):
    setup_paper_style()
    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)

    # Cumulative Communication
    xs_c, comms = get_xy(metrics, "comm_kb")
    if xs_c:
        cumulative_comms = np.cumsum(comms)
        plt.figure(figsize=(10, 6))
        plt.plot(xs_c, cumulative_comms, marker="o", label="Cumulative Communication (KB)", color='blue')
        plt.xlabel("Round")
        plt.ylabel("Cumulative Communication (KB)")
        plt.title("Cumulative Communication vs Round")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(Path(PLOTS_DIR) / "cumulative_comm_vs_round.pdf", dpi=300)
        plt.close()
        print(f"Saved {PLOTS_DIR}cumulative_comm_vs_round.pdf")

    # Cumulative Energy
    xs_e, energ = get_xy(metrics, "energy_mj")
    if xs_e:
        cumulative_energ = np.cumsum(energ)
        plt.figure(figsize=(10, 6))
        plt.plot(xs_e, cumulative_energ, marker="s", label="Cumulative Energy (mJ)", color='red')
        plt.xlabel("Round")
        plt.ylabel("Cumulative Energy (mJ)")
        plt.title("Cumulative Energy vs Round")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(Path(PLOTS_DIR) / "cumulative_energy_vs_round.pdf", dpi=300)
        plt.close()
        print(f"Saved {PLOTS_DIR}cumulative_energy_vs_round.pdf")


def plot_metrics(metrics):
    setup_paper_style() # Apply paper style to all plots
    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True) # Ensure plots directory exists

    # 1) Accuracy vs Round
    xs, accs = get_xy(metrics, "val_acc")
    if xs:
        plt.figure(figsize=(10, 6))
        plt.plot(xs, accs, marker="o")
        plt.xlabel("Round")
        plt.ylabel("Accuracy (%)")
        plt.title("Global Validation Accuracy vs Round")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(Path(PLOTS_DIR) / "accuracy_vs_round.pdf", dpi=300)
        plt.close()
        print(f"Saved {PLOTS_DIR}accuracy_vs_round.pdf")

    # 2) Loss vs Round
    xs, losses = get_xy(metrics, "val_loss")
    if xs:
        plt.figure(figsize=(10, 6))
        plt.plot(xs, losses, marker="o", color='red')
        plt.xlabel("Round")
        plt.ylabel("Avg Loss")
        plt.title("Global Validation Loss vs Round")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(Path(PLOTS_DIR) / "loss_vs_round.pdf", dpi=300)
        plt.close()
        print(f"Saved {PLOTS_DIR}loss_vs_round.pdf")

    # 3) Communication & Energy vs Round (Per Round)
    xs_c, comms = get_xy(metrics, "comm_kb")
    xs_e, energ = get_xy(metrics, "energy_mj")
    if xs_c or xs_e:
        plt.figure(figsize=(10, 6))
        if xs_c:
            plt.plot(xs_c, comms, marker="o", label="Communication (KB)", color='orange')
        if xs_e:
            plt.plot(xs_e, energ, marker="s", label="Energy (mJ)", color='brown')
        plt.xlabel("Round")
        plt.ylabel("Value")
        plt.title("Communication and Energy per Round")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(Path(PLOTS_DIR) / "comm_energy_per_round.pdf", dpi=300)
        plt.close()
        print(f"Saved {PLOTS_DIR}comm_energy_per_round.pdf")

    # 4) Rejected ratio vs Round
    xs_r, rej_ratio = get_xy(metrics, "rejected_ratio")
    if xs_r:
        plt.figure(figsize=(10, 6))
        plt.plot(xs_r, rej_ratio, marker="o", color='green')
        plt.xlabel("Round")
        plt.ylabel("Rejected / Total")
        plt.title("Defense Rejection Ratio vs Round")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(Path(PLOTS_DIR) / "rejected_ratio_vs_round.pdf", dpi=300)
        plt.close()
        print(f"Saved {PLOTS_DIR}rejected_ratio_vs_round.pdf")

    # 5) Anomaly mean & Reputation mean vs Round
    xs_a, anom = get_xy(metrics, "anomaly_mean")
    xs_rep, rep = get_xy(metrics, "reputation_mean")
    if xs_a or xs_rep:
        plt.figure(figsize=(10, 6))
        if xs_a:
            plt.plot(xs_a, anom, marker="o", label="Anomaly Mean", color='purple')
        if xs_rep:
            plt.plot(xs_rep, rep, marker="s", label="Reputation Mean", color='cyan')
        plt.xlabel("Round")
        plt.ylabel("Value")
        plt.title("Anomaly & Reputation vs Round")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(Path(PLOTS_DIR) / "anomaly_reputation_vs_round.pdf", dpi=300)
        plt.close()
    print(f"Saved {PLOTS_DIR}anomaly_reputation_vs_round.pdf")


def plot_comparative_comm_energy(experiment_logs, output_path, metric_key, title, ylabel):
    setup_paper_style()
    plt.figure(figsize=(10, 6))

    for name, log_file in experiment_logs.items():
        metrics = parse_log(log_file)
        xs, ys = get_xy(metrics, metric_key)
        if xs:
            plt.plot(xs, ys, label=name, marker='o', markersize=4)

    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{metric_key}_comparative_vs_round.pdf", dpi=300)
    plt.close()
    print(f"Saved {output_dir / f'{metric_key}_comparative_vs_round.pdf'}")


def main():
    metrics = parse_log(LOG_PATH)
    if not metrics:
        print("No rounds found – check LOG_PATH or regex patterns.")
        return

    print(f"Parsed {len(metrics)} rounds.")
    save_csv(metrics, CSV_PATH)
    plot_metrics(metrics)
    plot_cumulative_metrics(metrics) # Call the new cumulative plotting function

    comparative_logs = {
        "Model_1": "./logs/FedOpt/MNIST_lenet/iid/MNIST_lenet_iid_NoAttack_SystemAware_50_50_0.005_FedOpt/training_metrics.json",
        "Model_2": "./logs/FedOpt/MNIST_lenet/non-iid/MNIST_lenet_non-iid_AdaptiveAttack_SystemAware_5_5_0.01_FedOpt/training_metrics.json",
        "Model_3": "./logs/FedOpt/MNIST_lenet/non-iid/MNIST_lenet_non-iid_NoAttack_SystemAware_100_150_0.01_FedOpt/training_metrics.json",
    }
    
    valid_comparative_logs_comm = {}
    valid_comparative_logs_energy = {}
    for name, path in comparative_logs.items():
        if Path(path).exists():
            valid_comparative_logs_comm[name] = path
            valid_comparative_logs_energy[name] = path
        else:
            print(f"Warning: Log file not found for {name}: {path}. Skipping for comparative plots.")

    if valid_comparative_logs_comm:
        plot_comparative_comm_energy(valid_comparative_logs_comm, PLOTS_DIR, "comm_kb", "Comparative Communication Cost vs Round", "Communication (KB)")
    if valid_comparative_logs_energy:
        plot_comparative_comm_energy(valid_comparative_logs_energy, PLOTS_DIR, "energy_mj", "Comparative Energy Consumption vs Round", "Energy (mJ)")


if __name__ == "__main__":
    main()
