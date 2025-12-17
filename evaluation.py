# File: utils/evaluation.py
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, f1_score
import numpy as np

def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = F.cross_entropy(outputs, target)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100.0 * correct / total
    avg_loss = test_loss / len(test_loader)

    # Calculate F1 Score and Precision
    # Use 'weighted' average for multi-class classification to account for label imbalance
    precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)

    print(f"[Evaluation] Accuracy: {accuracy:.2f}% | Avg Loss: {avg_loss:.4f} | Precision: {precision:.4f} | F1 Score: {f1:.4f}")
    return accuracy, avg_loss, precision, f1
