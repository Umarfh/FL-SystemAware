import os
import torch
import yaml
import argparse
from types import SimpleNamespace
from global_utils import frac_or_int_to_int

# Ensure aggregators and attackers are fully initialized before use
import aggregators
import attackers
from fl.models import all_models
from fl.algorithms import all_algorithms


def read_args():
    """
    1. parse command line arguments for configuration path and possible arguments.
    2. load configurations to `config` from the provided YAML file.
    3. load data configurations from the `dataset_config.yaml` file, while overriding `epochs` and learning rate `lr`.
    4. override the `config` with command line arguments if provided.
    return the `config` object with all configurations.
    """
    parser = argparse.ArgumentParser(
        description="Poisoning attacks and defenses in Federated Learning")
    parser.add_argument('-config', '--config', type=str,
                        required=True, help='Path to the YAML configuration file')
    # command line arguments if provided
    parser.add_argument('-b', '--benchmark', default=False, type=bool,
                        help='Run all combinations of attacks and defenses')
    parser.add_argument('-e', '--epochs', type=int)
    parser.add_argument('-seed', '--seed', type=int)
    parser.add_argument('-alg', '--algorithm', choices=all_algorithms)
    parser.add_argument('-opt', '--optimizer', choices=['SGD', 'Adam'],
                        help='optimizer for training')
    parser.add_argument('-lr_scheduler', '--lr_scheduler', type=str,
                        help='lr_scheduler for training')
    parser.add_argument('-milestones', '--milestones', type=int, nargs="+",
                        help='milestone for learning rate scheduler')
    parser.add_argument('-num_clients', '--num_clients', type=int,
                        help='number of participating clients')
    parser.add_argument('-bs', '--batch_size', type=int,
                        help='batch_size')
    parser.add_argument('-lr', '--learning_rate',
                        type=float, help='initial learning rate')
    parser.add_argument('-le', '--local_epochs', type=int,
                        help='local global_epoch')
    parser.add_argument('-model', '--model', choices=all_models)
    parser.add_argument('-data', '--dataset',
                        choices=['MNIST', 'FashionMNIST', 'CIFAR10', 'CINIC10', 'CIFAR100', 'EMNIST'])
    parser.add_argument('-dtb', '--distribution',
                        choices=['iid', 'class-imbalanced_iid', 'non-iid', 'pat', 'imbalanced_pat'])
    parser.add_argument('-dirichlet_alpha', '--dirichlet_alpha', type=float,
                        help='smaller alpha for drichlet distribution, stronger heterogeneity, 0.1 0.5 1 5 10, normally use 0.5')
    parser.add_argument('-im_iid_gamma', '--im_iid_gamma', type=float,
                        help='smaller alpha for class imbalanced distribution, stronger heterogeneity, 0.05, 0.1, 0.5')

    # attacks and defenses settings
    all_attacks = ['NoAttack'] + \
        attackers.model_poisoning_attacks + attackers.data_poisoning_attacks + ['AdaptiveAttack']
    parser.add_argument('-att', '--attack',
                        choices=all_attacks, help="Attacks options")
    parser.add_argument('-attack_start_epoch', '--attack_start_epoch',
                        type=int, help="the attack start epoch")
    parser.add_argument('-attparam', '--attparam', type=float,
                        help='scale for omniscient model poisoning attack, IPM,ALIE,MinMax,MinSum,Fang')
    parser.add_argument('-def', '--defense',
                        choices=aggregators.all_aggregators, help="Defenses options") # Use aggregators.all_aggregators
    parser.add_argument('-num_adv', '--num_adv', type=float,
                        help='the proportion (float < 1) or number (int>1) of adversaries')
    parser.add_argument('-o', '--output', type=str,
                        help='output file for results')
    # poison settings
    parser.add_argument('-prate', '--poisoning_ratio',
                        help='poisoning portion (float, range from 0 to 1, default: 0.1)')
    parser.add_argument('--target_label', type=int,
                        help='The No. of target label for backdoored images (int, range from 0 to 10, default: 6)')
    parser.add_argument('--trigger_path', help='Trigger Path')
    parser.add_argument('--trigger_size', type=int,
                        help='Trigger Size (int, default: 5)')
    parser.add_argument('-gidx', '--gpu_idx', type=int, default=3,
                        help='Index of GPU (int, default: 3, choice: 0, 1, 2, 3...)')
    parser.add_argument("--num_shards", type=int, default=200,
                    help="number of shards for shard-based non-iid")
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='Enable verbose logging for defense pipeline') # Added verbose argument
    parser.add_argument('-esp', '--early_stopping_patience', type=int,
                        help='Number of epochs to wait for improvement before early stopping')


    # override attack_params or defense_params with dict string
    parser.add_argument(
        '-defense_params', '--defense_params', type=str, help='Override defense parameters')
    parser.add_argument(
        '-attack_params', '--attack_params', type=str, help='Override attack parameters')
    
    import sys
    print(f"DEBUG: sys.argv before parse_args: {sys.argv}") # Debug print
    cli_args = parser.parse_args()
    

    # load configurations from yaml file if provided
    args = SimpleNamespace()  # compatible with argparse.Namespace
    if cli_args.config:
        args = read_yaml(cli_args.config, all_attacks, aggregators.all_aggregators)
    return args, cli_args


def read_yaml(filename, all_attacks_list, all_aggregators_list):
    # read configurations from yaml file to args dict object
    with open(filename.strip(), 'r', encoding='utf-8') as file:
        args_dict = yaml.load(file, Loader=yaml.FullLoader)
    
    # If 'dataset' is a dictionary, extract its 'name'
    if 'dataset' in args_dict and isinstance(args_dict['dataset'], dict):
        args_dict['dataset'] = args_dict['dataset'].get('name', 'Unknown') # Default to 'Unknown' if name is missing

    # Set a default early_stopping_patience if not already present in args_dict
    if 'early_stopping_patience' not in args_dict:
        args_dict['early_stopping_patience'] = 10 # Default patience
    
    # If 'algorithm' is a dictionary, extract its 'name'
    if 'algorithm' in args_dict and isinstance(args_dict['algorithm'], dict):
        args_dict['algorithm'] = args_dict['algorithm'].get('name', 'UnknownAlgorithm') # Default to 'UnknownAlgorithm'
    
    # If 'model' is a dictionary, extract its 'name'
    if 'model' in args_dict and isinstance(args_dict['model'], dict):
        args_dict['model'] = args_dict['model'].get('name', 'UnknownModel') # Default to 'UnknownModel'
    
    # If 'defense' is a dictionary, extract its 'name' and 'defense_params'
    if 'defense' in args_dict:
        if isinstance(args_dict['defense'], dict):
            args_dict['defense_params'] = args_dict['defense'].get('defense_params')
            args_dict['defense'] = args_dict['defense'].get('name', 'UnknownDefense')
        else: # It's a string
            args_dict['defense_params'] = None
    
    # If 'attack' is a dictionary, extract its 'name' and 'attack_params'
    if 'attack' in args_dict:
        if isinstance(args_dict['attack'], dict):
            args_dict['attack_params'] = args_dict['attack'].get('attack_params')
            args_dict['attack'] = args_dict['attack'].get('name', 'UnknownAttack')
        else: # It's a string
            args_dict['attack_params'] = None

    args = SimpleNamespace(**args_dict)
    
    # Ensure 'attack' and 'defense' attributes exist, even if None
    ensure_attr(args, 'attack')
    ensure_attr(args, 'defense')

    # Map 'dtb' from config to 'distribution' in args
    if hasattr(args, 'dtb'):
        args.distribution = args.dtb
    else:
        args.distribution = 'iid' # Default if not specified

    # Initialize num_adv based on attack configuration
    if hasattr(args, 'attack_type') and args.attack_type != 'none' and hasattr(args, 'num_malicious'):
        args.num_adv = args.num_malicious
    else:
        args.num_adv = 0 # Default to 0 if no attack or num_malicious not specified

    # Map 'lr' from config to 'learning_rate' in args
    if hasattr(args, 'lr'):
        args.learning_rate = args.lr
    else:
        args.learning_rate = 0.01 # Default if not specified

    # Add all_attacks and all_aggregators to args for override_args to use
    args.attacks = all_attacks_list
    args.defenses = all_aggregators_list
    
    return args


def override_args(args, cli_args):
    """
    1. fill the attack and defense parameters with default if not provided.
    2. override the arguments with provided command line arguments if possible.
    if attack and defense are provided:
        if their corresponding parameters provided:
            override them with the provided parameters
        else:
            override them with default attack parameters
    Args:
        args: the configuration object readin from the yaml file
        cli_args: the command line arguments
    """
    # fill the attack and defense parameters with default
    for param_type in ['attack', 'defense']:
        if not hasattr(args, f"{param_type}_params"):
            # The lists args.attacks and args.defenses contain strings, not dictionaries.
            # So, we cannot use i[param_type].
            # Instead, we check if the current param_type (e.g., 'attack') matches any string in the list.
            # If no specific parameters are defined for this attack/defense, set params to None.
            if getattr(args, param_type) in getattr(args, f"{param_type}s"):
                setattr(args, f"{param_type}_params", None)

    # override parameters
    # if only attack or defense is provided, set their corresponding params to default
    for key, value in vars(cli_args).items():
        if key in ['config', 'attack', 'defense', 'attack_params', 'defense_params']:
            continue
        if value is not None:
            setattr(args, key, value)

            print(f"Warning: Overriding {key} with {value}")

    # override attack, defense, attack_params, defense_params
    for param_type in ['attack', 'defense']:
        if eval(f"cli_args.{param_type}"):  # if not None
            setattr(args, param_type, eval(f"cli_args.{param_type}"))
            # if attack_params or defense_params is provided by cli_args, override the corresponding params
            if eval(f"cli_args.{param_type}_params"):
                setattr(args, f'{param_type}_params',
                        eval(f"cli_args.{param_type}_params"))
            else:
                # if not provided, set the params to default
                for i in eval(f"args.{param_type}s"):
                    if i[param_type] == eval(f"args.{param_type}"):
                        setattr(args, f"{param_type}_params",
                                i.get(f"{param_type}_params"))
                        break


def benchmark_preprocess(args):
    for attack_i in args.attacks:
        for defense_j in args.defenses:
            args.attack, args.attack_params = attack_i['attack'], attack_i.get(
                'attack_params')
            args.defense, args.defense_params = defense_j['defense'], defense_j.get(
                'defense_params')
            single_preprocess(args)
            if os.path.exists(args.output):
                print(f"File {args.output.split('/')[-1]} exists, skip")
                continue
            print(
                f"Running {args.attack} with {args.defense} under {args.distribution}")


def single_preprocess(args):
    # load dataset configurations, also include learning rate and epochs
    with open("./configs/dataset_config.yaml", 'r') as file:
        dataset_config = yaml.load(file, Loader=yaml.FullLoader)
    for key, value in dataset_config[args.dataset].items():
        if key in ['mean', 'std']:
            value = eval(value)
        setattr(args, key, value)

    # Set a default learning rate if not already present
    if not hasattr(args, 'lr'):
        args.lr = 0.01 # Default learning rate
        print(f"Warning: 'lr' not found in config, setting to default: {args.lr}")

    # Set a default epochs if not already present
    if not hasattr(args, 'epochs'):
        args.epochs = 500 # Default epochs
        print(f"Warning: 'epochs' not found in config, setting to default: {args.epochs}")

    # Set a default num_clients if not already present
    if not hasattr(args, 'num_clients'):
        args.num_clients = 200 # Default num_clients
        print(f"Warning: 'num_clients' not found in config, setting to default: {args.num_clients}")

    # preprocess the arguments
    # Priority: CUDA > MPS (MacOS) > CPU
    if torch.cuda.is_available():
        # Ensure args.gpu_idx is a list, even if a single int is provided
        if isinstance(args.gpu_idx, int):
            args.gpu_idx = [args.gpu_idx]

        # Validate the specified GPU index
        if args.gpu_idx[0] < torch.cuda.device_count():
            device = torch.device(f"cuda:{args.gpu_idx[0]}")
        else:
            print(f"Warning: Specified GPU index {args.gpu_idx[0]} is invalid. "
                  f"Only {torch.cuda.device_count()} CUDA devices available. "
                  f"Falling back to cuda:0 if available, otherwise CPU.")
            if torch.cuda.device_count() > 0:
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    args.device = device
    args.num_adv = frac_or_int_to_int(args.num_adv, args.num_clients)
    # Set num_attackers for orchestrator logging
    args.num_attackers = args.num_adv

    # ensure attack_params and defense_params attributes exist. when there is no params, set it to None.
    ensure_attr(args, 'attack_params')
    ensure_attr(args, 'defense_params')

    # Ensure args.aggregator is set based on args.defense
    if hasattr(args, 'defense'):
        args.aggregator = args.defense
    else:
        args.aggregator = "FedAvg" # Default if no defense is specified

    # generate output path if not provided
    # Ensure output is a directory path, not a file path, for metrics.save
    output_filename = f'{args.dataset}_{args.model}_{args.distribution}_{args.attack}_{args.defense}_{args.epochs}_{args.num_clients}_{args.learning_rate}_{args.algorithm}'
    args.output = f'./logs/{args.algorithm}/{args.dataset}_{args.model}/{args.distribution}/{output_filename}'

    # check output path, if exists, skip, otherwise create the directories
    os.makedirs(args.output, exist_ok=True) # Create the directory
    return args


def ensure_attr(obj, attr_name):
    """
    set attr_name of obj to None if it does not exist
    """
    if not hasattr(obj, attr_name):
        setattr(obj, attr_name, None)
