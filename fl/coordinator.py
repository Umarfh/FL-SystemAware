import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict
from collections import OrderedDict
from .client import Client
from attackers import get_attacker_handler
from datapreprocessor.data_utils import subset_by_idx
from attackers import data_poisoning_attacks, hybrid_attacks
import copy # Import copy for deepcopy

def get_dataloader(dataset, batch_size=64, train=False, num_workers=0, pin_memory=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, pin_memory=pin_memory)

def init_clients(args, client_indices, train_dataset, test_dataset):
    clients = []
    for worker_id in range(args.num_clients):
        # for attacker, if the attack type is not model poisoning attack, use the default client class. For data poisoning attacks, it's already handled in the client class.
        # for benign clients, use the default client class
        if args.attack == "NoAttack":
            """
            For NoAttack scenario, use client class, and ignore args.num_adv
            """
            client_obj = Client
        else:
            if args.num_adv == 0:
                raise AssertionError(
                    "Attack {args.attack} specified, but attackers set to 0.")
            client_obj = Client if worker_id >= args.num_adv else get_attacker_handler(
                args.attack)
        local_dataset = subset_by_idx(
            train_dataset, client_indices[worker_id])
        tmp_client = client_obj(args=args, worker_id=worker_id,
                                train_dataset=local_dataset, test_dataset=test_dataset)
        clients.append(tmp_client)
    return clients


def set_fl_algorithm(args, the_server, clients):
    """set the federated learning algorithm for the server and clients. If the algorithm type is not specified in arguments, use the default algorithm type of the server.

    Args:
        the_server (Server): server object
        clients (Client): a list of client objects
        algorithm (str): federated learning algorithm types

    Raises:
        ValueError: No specified or default algorithm type can be used
    """
    if args.algorithm:
        alg_type = args.algorithm
    elif hasattr(the_server, 'algorithm'):
        args.algorithm = the_server.algorithm
    else:
        raise ValueError(
            "No specified algorithm or default algorithm type of the server. Please specify an algorithm type, with `--algorithm`")

    for client in clients:
        if hasattr(client, 'set_algorithm'):
            client.set_algorithm(alg_type)


def evaluate(server, test_dataset, args, global_epoch):
    """
    Evaluate server.global_model on test_dataset.
    Returns a dict with 'test_acc' and 'test_loss' and some extra metrics.
    """
    model = server.global_model
    model.eval()
    device = getattr(args, "device", "cpu")
    test_loader = get_dataloader(test_dataset, batch_size=getattr(args, "batch_size", 64), train=False, num_workers=getattr(args, "num_workers", 0), pin_memory=False)

    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            out = model(data)
            loss = F.cross_entropy(out, target, reduction='sum')
            test_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += int((preds == target).sum().item())
            total += target.size(0)

    test_loss = test_loss / (total + 1e-9)
    test_acc = correct / (total + 1e-9)

    # optional: add precision, f1 placeholders (implement if needed)
    metrics = {
        "test_acc": test_acc,
        "test_loss": test_loss,
        "precision": None,
        "f1": None
    }
    # log via server.logger if present
    try:
        server.logger.info(f"[Evaluation] Accuracy: {test_acc*100:.2f}% | Avg Loss: {test_loss:.4f}")
    except Exception:
        pass

    return metrics
