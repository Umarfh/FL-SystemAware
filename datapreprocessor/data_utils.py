import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_transform(args):
    dataset_name = args.dataset.lower()
    
    if dataset_name == "mnist":
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return trans, test_trans

    elif dataset_name == "cifar10":
        # Training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
        
        # Test transforms (no augmentation)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
        return train_transform, test_transform

    else:
        raise ValueError(f"Dataset {args.dataset} not implemented yet")


def load_data(args):
    """Load dataset and return train/test sets"""
    trans, test_trans = get_transform(args)
    dataset_name = args.dataset.lower()
    
    if dataset_name == "mnist":
        train_dataset = datasets.MNIST(root='./data', train=True,
                                       download=True, transform=trans)
        test_dataset = datasets.MNIST(root='./data', train=False,
                                      download=True, transform=test_trans)

    elif dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=trans)
        test_dataset = datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_trans)
    else:
        raise ValueError(f"Dataset {args.dataset} not implemented yet")

    return train_dataset, test_dataset


def split_dataset_non_iid(dataset, num_clients, num_shards=200):
    """
    Split a dataset in a non-IID fashion using the shard method.
    Returns a list of indices per client.
    """
    # Get labels
    targets = np.array(dataset.targets)
    data_size = len(dataset)
    
    # sort indices by label
    idxs = np.argsort(targets)
    shards_size = data_size // num_shards
    shards = [idxs[i*shards_size:(i+1)*shards_size] for i in range(num_shards)]
    
    # assign shards to clients
    client_shards = {i: [] for i in range(num_clients)}
    shard_idxs = np.random.permutation(num_shards)
    
    for i, shard_id in enumerate(shard_idxs):
        client_id = i % num_clients
        client_shards[client_id].extend(shards[shard_id])
    
    return client_shards


def get_dataloader(dataset, indices=None, batch_size=32, train_flag=True):
    if indices is not None:
        subset = Subset(dataset, indices)
    else:
        subset = dataset

    shuffle = train_flag
    loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle)
    return loader


def dataset_class_indices(dataset):
    """
    Returns a dictionary of class indices for a dataset.
    Keys are class labels, values are lists of indices.
    """
    class_indices = {i: [] for i in range(len(dataset.classes))}
    for i, target in enumerate(dataset.targets):
        class_indices[target].append(i)
    return class_indices


def subset_by_idx(dataset, indices):
    """
    Returns a Subset of the dataset given a list of indices.
    """
    return Subset(dataset, indices)


class Partition(object):
    """ Dataset partitioning helper """
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


def split_dataset_iid(dataset, num_clients):
    """
    Split a dataset in an IID fashion.
    Returns a list of indices per client.
    """
    data_size = len(dataset)
    client_data_size = data_size // num_clients
    
    idxs = np.random.permutation(data_size)
    
    client_indices = {i: [] for i in range(num_clients)}
    for i in range(num_clients):
        client_indices[i] = idxs[i * client_data_size: (i + 1) * client_data_size]
    
    return client_indices


def split_dataset(dataset, num_clients, distribution="iid", num_shards=200):
    """
    Splits the dataset among clients based on the specified distribution.
    """
    if distribution == "iid":
        return split_dataset_iid(dataset, num_clients)
    elif distribution == "non-iid":
        return split_dataset_non_iid(dataset, num_clients, num_shards)
    else:
        raise ValueError(f"Distribution {distribution} not supported.")
