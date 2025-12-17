import re
import csv
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import plot_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_utils import setup_paper_style

LOG_PATH = "logs/fl_training.log"
PLOTS_DIR = "plots/" # Define a dedicated directory for plots
CSV_PATH = Path(PLOTS_DIR) / "metrics.csv" # Save CSV in the plots directory

def parse_log(path):
    metrics = []
    current = {}

    expect_anomaly_mean = False
    expect_reputation_mean = False

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # --- New round (Metrics block) ---
            m_round = re.search(r"\[Metrics\]\s*Round\s+(\d+):", line)
            if m_round:
                # store previous round if exists
                if "round" in current:
                    metrics.append(current)
                    current = {}

                current["round"] = int(m_round.group(1))
                continue

            # --- Global evaluation (accuracy + loss) ---
            # Example: [Server] [Evaluation] Accuracy: 97.81% | Avg Loss: 0.1339
            m_eval = re.search(
                r"Accuracy:\s*([\d.]+)%\s*\|\s*Avg Loss:\s*([\d.]+)", line
            )
            if m_eval and "round" in current:
                current["val_acc"] = float(m_eval.group(1))
                current["val_loss"] = float(m_eval.group(2))
                continue

            # --- Communication ---
            # Example: [Server]   Communication: 5206.17 KB
            m_comm = re.search(r"Communication:\s*([\d.]+)\s*KB", line)
            if m_comm and "round" in current:
                current["comm_kb"] = float(m_comm.group(1))
                continue

            # --- Energy ---
            # Example: [Server]   Energy:        260.309 mJ (coeff=0.05)
            m_energy = re.search(r"Energy:\s*([\d.]+)\s*mJ", line)
            if m_energy and "round" in current:
                current["energy_mj"] = float(m_energy.group(1))
                continue

            # --- Defense rejected count ---
            # Example: [Server]   Defense rejected: 6/30 updates
            m_rej = re.search(r"Defense rejected:\s*(\d+)\s*/\s*(\d+)\s*updates", line)
            if m_rej and "round" in current:
                rejected = int(m_rej.group(1))
                total = int(m_rej.group(2))
                current["rejected"] = rejected
                current["total_updates"] = total
                if total > 0:
                    current["rejected_ratio"] = rejected / total
                continue

            # --- Anomaly mean ---
            # Block structure:
            # [Server] Anomaly Scores:
            # [Server]   Mean:   0.2914
            if "Anomaly Scores:" in line and "round" in current:
                expect_anomaly_mean = True
                continue

            if expect_anomaly_mean and "Mean:" in line and "round" in current:
                m_anom = re.search(r"Mean:\s*([\d.]+)", line)
                if m_anom:
                    current["anomaly_mean"] = float(m_anom.group(1))
                expect_anomaly_mean = False
                continue

            # --- Reputation mean ---
            # [Server] Reputations:
            # [Server]   Mean:   0.402
            if "Reputations:" in line and "round" in current:
                expect_reputation_mean = True
                continue

            if expect_reputation_mean and "Mean:" in line and "round" in current:
                m_rep = re.search(r"Mean:\s*([\d.]+)", line)
                if m_rep:
                    current["reputation_mean"] = float(m_rep.group(1))
                expect_reputation_mean = False
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


def main():
    metrics = parse_log(LOG_PATH)
    if not metrics:
        print("No rounds found – check LOG_PATH or regex patterns.")
        return

    print(f"Parsed {len(metrics)} rounds.")
    save_csv(metrics, CSV_PATH)
    plot_metrics(metrics)
    plot_cumulative_metrics(metrics) # Call the new cumulative plotting function


if __name__ == "__main__":
    main()
