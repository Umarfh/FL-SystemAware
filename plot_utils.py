import re
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend, good for scripts
import matplotlib.pyplot as plt
from pathlib import Path # Import Path
import json # Import json

import numpy as np


def setup_paper_style():
    plt.style.use('seaborn-v0_8-paper') # A good starting point for paper-like plots
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'savefig.dpi': 300, # High resolution for saving
        'savefig.format': 'pdf', # Save as PDF for vector graphics
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Palatino', 'serif'],
        'text.usetex': False, # Set to True if LaTeX is installed for better text rendering
    })

def parse_logs(filename):
    plt.clf()
    # read log file
    with open(filename, 'r') as f:
        content = json.load(f) # Load JSON content

    epochs, accs, losses, asrs, asr_losses = [], [], [], [], []
    round_times, active_clients, stragglers, dropouts = [], [], [], []
    
    for round_data in content.get('rounds', []):
        epochs.append(round_data.get('round'))
        accs.append(round_data.get('test_acc'))
        losses.append(round_data.get('test_loss'))
        asrs.append(round_data.get('asr')) # Assuming 'asr' might be present in some logs
        asr_losses.append(round_data.get('asr_loss')) # Assuming 'asr_loss' might be present

        round_times.append(round_data.get('round_time'))
        active_clients.append(round_data.get('active_clients'))
        stragglers.append(round_data.get('stragglers'))
        dropouts.append(round_data.get('dropouts'))

    return epochs, accs, losses, asrs, asr_losses, round_times, active_clients, stragglers, dropouts


def plot_accuracy(epochs, accs, output_path):
    setup_paper_style()
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accs, label='Test Accuracy', marker='o')

    plt.xlabel('Round Number')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy vs. Round Number')
    plt.legend()
    plt.grid(True)
    
    # Ensure the output directory exists
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / "accuracy_plot.pdf")
    plt.close() # Close the plot to free memory


def plot_loss(epochs, losses, output_path):
    setup_paper_style()
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, label='Test Loss', marker='o', color='red')

    plt.xlabel('Round Number')
    plt.ylabel('Loss')
    plt.title('Test Loss vs. Round Number')
    plt.legend()
    plt.grid(True)
    
    # Ensure the output directory exists
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / "loss_plot.pdf")
    plt.close() # Close the plot to free memory


def plot_asr(epochs, asrs, output_path):
    setup_paper_style()
    # Filter out None values for ASR if they exist
    valid_asrs = [(e, a) for e, a in zip(epochs, asrs) if a is not None]
    if not valid_asrs:
        print("No ASR data to plot.")
        return

    epochs_filtered, asrs_filtered = zip(*valid_asrs)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs_filtered, asrs_filtered, label='Attack Success Rate', marker='o', color='green')

    plt.xlabel('Round Number')
    plt.ylabel('ASR')
    plt.title('Attack Success Rate vs. Round Number')
    plt.legend()
    plt.grid(True)
    
    # Ensure the output directory exists
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / "asr_plot.pdf")
    plt.close() # Close the plot to free memory


def plot_round_time(epochs, round_times, output_path):
    setup_paper_style()
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, round_times, label='Round Time', marker='o', color='purple')

    plt.xlabel('Round Number')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time per Round')
    plt.legend()
    plt.grid(True)
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / "round_time_plot.pdf")
    plt.close()


def plot_rejected_accepted_updates(epochs, active_clients, dropouts, output_path):
    setup_paper_style()
    plt.figure(figsize=(10, 6))
    
    # Assuming 'active_clients' are accepted and 'dropouts' are rejected
    # You might need to adjust this based on the exact meaning in your logs
    plt.plot(epochs, active_clients, label='Accepted Updates', marker='o', color='blue')
    plt.plot(epochs, dropouts, label='Rejected Updates (Dropouts)', marker='x', color='red')

    plt.xlabel('Round Number')
    plt.ylabel('Number of Updates')
    plt.title('Rejected vs. Accepted Updates per Round')
    plt.legend()
    plt.grid(True)
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / "rejected_accepted_updates_plot.pdf")
    plt.close()


def plot_comparative_accuracy(experiment_logs, output_path):
    setup_paper_style()
    accuracies = []
    labels = []

    for name, log_file in experiment_logs.items():
        with open(log_file, 'r') as f:
            content = json.load(f)
        final_accuracy = content.get('final_test_accuracy', 0.0)
        accuracies.append(final_accuracy)
        labels.append(name)

    plt.figure(figsize=(12, 7))
    plt.bar(labels, accuracies, color='skyblue')
    plt.xlabel('Experiment Configuration')
    plt.ylabel('Final Test Accuracy')
    plt.title('Comparative Final Test Accuracy')
    plt.ylim(0, 1) # Assuming accuracy is between 0 and 1
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "comparative_accuracy_bar_chart.pdf")
    plt.close()


def plot_label_distribution(train_data, client_idcs, n_clients, dataset, distribution):
    setup_paper_style() # Apply paper style to label distribution as well
    titleid_dict = {"iid": "Balanced_IID", "class-imbalanced_iid": "Class-imbalanced_IID",
                    "non-iid": "Quantity-imbalanced_Dirichlet_Non-IID", "pat": "Balanced_Pathological_Non-IID", "imbalanced_pat": "Quantity-imbalanced_Pathological_Non-IID"}
    dataset = "CIFAR-10" if dataset == "CIFAR10" else dataset
    title_id = dataset + " " + titleid_dict[distribution]
    xy_type = "client_label"  # 'label_client'
    # plt.rcParams['font.size'] = 14  # set global fontsize - these are now handled by setup_paper_style
    # set the direction of xtick toward inside
    # plt.rcParams['xtick.direction'] = 'in'
    # plt.rcParams['ytick.direction'] = 'in'
    # plt.rcParams['font.sans-serif'] = ['Times New Roman']
    labels = train_data.targets
    n_classes = labels.max()+1
    plt.figure(figsize=(12, 8))
    if xy_type == "client_label":
        label_distribution = [[] for _ in range(n_classes)]
        for c_id, idc in enumerate(client_idcs):
            for idx in idc:
                label_distribution[labels[idx]].append(c_id)

        plt.hist(label_distribution, stacked=True,
                 bins=np.arange(-0.5, n_clients + 1.5, 1),
                 label=range(n_classes), rwidth=0.5, zorder=10)
        plt.xticks(np.arange(n_clients), ["%d" %
                                          c_id for c_id in range(n_clients)])
        plt.xlabel("Client ID") # Fontsize handled by style
    elif xy_type == "label_client":
        plt.hist([labels[idc]for idc in client_idcs], stacked=True,
                 bins=np.arange(min(labels)-0.5, max(labels) + 1.5, 1),
                 label=["Client {}".format(i) for i in range(n_clients)],
                 rwidth=0.5, zorder=10)
        plt.xticks(np.arange(n_classes), train_data.classes)
        plt.xlabel("Label type") # Fontsize handled by style

    plt.ylabel("Number of Training Samples") # Fontsize handled by style
    plt.title(f"{title_id} Label Distribution Across Clients") # Fontsize handled by style
    rotation_degree = 45 if n_clients > 30 else 0
    plt.xticks(rotation=rotation_degree) # Fontsize handled by style
    plt.legend(loc="best").set_zorder(100) # Fontsize handled by style
    plt.grid(linestyle='--', axis='y', zorder=0)
    plt.tight_layout()
    plt.savefig(Path(output_path) / f"{title_id}_label_dtb.pdf",
                format='pdf', bbox_inches='tight')


def plot_anomaly_scores_distribution(anomaly_scores_data, output_path):
    setup_paper_style()
    if not anomaly_scores_data:
        print("No anomaly scores data to plot.")
        return

    # Assuming anomaly_scores_data is a list of lists, where each inner list is scores for a round
    plt.figure(figsize=(12, 7))
    plt.boxplot(anomaly_scores_data, patch_artist=True, medianprops={'color': 'black'})
    plt.xlabel('Round Number')
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Scores Distribution per Round')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "anomaly_scores_distribution_plot.pdf")
    plt.close()


def plot_communication_cost(epochs, communication_costs, output_path):
    setup_paper_style()
    if not communication_costs:
        print("No communication cost data to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, communication_costs, label='Communication Cost', marker='o', color='orange')
    plt.xlabel('Round Number')
    plt.ylabel('Total Data Transmitted (MB)')
    plt.title('Communication Cost vs. Round Number')
    plt.legend()
    plt.grid(True)

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "communication_cost_plot.pdf")
    plt.close()


def plot_energy_consumption(epochs, energy_consumptions, output_path):
    setup_paper_style()
    if not energy_consumptions:
        print("No energy consumption data to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, energy_consumptions, label='Energy Consumption', marker='o', color='brown')
    plt.xlabel('Round Number')
    plt.ylabel('Energy (mJ)')
    plt.title('Energy Consumption vs. Round Number')
    plt.legend()
    plt.grid(True)

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "energy_consumption_plot.pdf")
    plt.close()


if __name__ == "__main__":
    log_file = "./logs/FedOpt/MNIST_lenet/iid/MNIST_lenet_iid_AdaptiveAttack_SystemAware_100_50_0.01_FedOpt/experiment_metrics.json"
    output_dir = "./plots/" # Changed output directory to a new 'plots' folder

    # Example usage:
    epochs, accs, losses, asrs, asr_losses, round_times, active_clients, stragglers, dropouts = parse_logs(log_file)
    plot_accuracy(epochs, accs, output_dir)
    plot_loss(epochs, losses, output_dir)
    plot_asr(epochs, asrs, output_dir)
    plot_round_time(epochs, round_times, output_dir)
    plot_rejected_accepted_updates(epochs, active_clients, dropouts, output_dir)

    # Placeholder for data not found in experiment_metrics.json
    # Anomaly Scores Distribution: Requires per-client anomaly scores per round.
    # Communication Cost: Requires total data transmitted per round.
    # Energy Consumption: Requires energy consumption per round.
    print("Note: Anomaly Scores Distribution, Communication Cost, and Energy Consumption plots require additional data not present in the current experiment_metrics.json.")
    print("Please ensure these metrics are logged in the JSON file for these plots to be generated.")

    # Example for comparative accuracy (this part would typically be run separately or with a different main logic)
    comparative_log_files = {
        "FedAvg_NoDefense": "./logs/FedAvg/CIFAR10_SimpleCNN/iid/CIFAR10_SimpleCNN_iid_None_None_10_10_0.01_FedAvg/experiment_metrics.json",
        "FedOpt_NoDefense": "./logs/FedOpt/CIFAR10_SimpleCNN/iid/CIFAR10_simplecnn_iid_NoAttack_None_10_20_0.01_FedOpt/training_metrics.json", # This file is named training_metrics.json, not experiment_metrics.json
        "FedOpt_SystemAware": "./logs/FedOpt/MNIST_lenet/iid/MNIST_lenet_iid_AdaptiveAttack_SystemAware_100_50_0.01_FedOpt/experiment_metrics.json"
    }
    # plot_comparative_accuracy(comparative_log_files, output_dir) # Uncomment to run comparative plot
