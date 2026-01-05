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
    parser.add_argument('-b', '--benchmark', default=False, action='store_true',
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
    
    # New arguments for dynamic communication and energy costs
    parser.add_argument('-alpha', '--alpha_scaling_factor', type=float, default=0.1,
                        help='Scaling factor alpha for anomaly score in communication cost')
    parser.add_argument('-beta', '--beta_scaling_factor', type=float, default=1.0,
                        help='Scaling factor beta for communication cost in energy consumption')
    parser.add_argument('-e0', '--energy_offset', type=float, default=10.0,
                        help='Base energy consumption offset E0')

    cli_args = parser.parse_args()
    

    # load configurations from yaml file if provided
    args = SimpleNamespace()  # compatible with argparse.Namespace
    if cli_args.config:
        args = read_yaml(cli_args.config, all_attacks, aggregators.all_aggregators)
    return args, cli_args


def read_yaml(filename, all_attacks_list, all_aggregators_list):
    with open(filename.strip(), 'r', encoding='utf-8') as file:
        args_dict = yaml.safe_load(file) # Use safe_load for security

    # Initialize with default values or top-level values
    final_args = SimpleNamespace()

    # Set default for participation_rate early
    final_args.participation_rate = args_dict.get('participation_rate', 0.6)

    # Handle top-level simple parameters
    final_args.epochs = args_dict.get('epochs', 100)
    final_args.num_clients = args_dict.get('num_clients', 200)
    final_args.batch_size = args_dict.get('batch_size', 32)
    final_args.learning_rate = args_dict.get('learning_rate', 0.01)
    final_args.local_epochs = args_dict.get('local_epochs', 5)
    final_args.seed = args_dict.get('seed', 4)
    final_args.momentum = args_dict.get('momentum', 0.9)
    final_args.weight_decay = args_dict.get('weight_decay', 5.0e-4)
    final_args.lr_scheduler = args_dict.get('lr_scheduler', 'StepLR')
    final_args.milestones = args_dict.get('milestones', [2, 4])
    final_args.gamma = args_dict.get('gamma', 0.95)
    final_args.im_iid_gamma = args_dict.get('im_iid_gamma', 0.01)
    final_args.tail_cls_from = args_dict.get('tail_cls_from', 4)
    final_args.dirichlet_alpha = args_dict.get('dirichlet_alpha', 0.5)
    final_args.cache_partition = args_dict.get('cache_partition', False)
    final_args.gpu_idx = args_dict.get('gpu_idx', [0])
    final_args.num_workers = args_dict.get('num_workers', 0)
    final_args.record_time = args_dict.get('record_time', False)
    final_args.num_adv = args_dict.get('num_adv', 0)
    final_args.num_shards = args_dict.get('num_shards', 200)
    final_args.early_stopping_patience = args_dict.get('early_stopping_patience', 10)
    final_args.verbose = args_dict.get('verbose', False) # Top-level verbose

    # Handle top-level complex parameters (dictionaries)
    # Dataset
    dataset_config = args_dict.get('dataset', 'MNIST')
    if isinstance(dataset_config, dict):
        final_args.dataset = dataset_config.get('name', 'MNIST')
        final_args.distribution = dataset_config.get('distribution', 'non-iid')
    else:
        final_args.dataset = dataset_config
        final_args.distribution = args_dict.get('distribution', 'non-iid') # Fallback to top-level distribution

    # Model
    model_config = args_dict.get('model', 'lenet')
    if isinstance(model_config, dict):
        final_args.model = model_config.get('name', 'lenet')
    else:
        final_args.model = model_config

    # Algorithm
    algorithm_config = args_dict.get('algorithm', 'FedOpt')
    if isinstance(algorithm_config, dict):
        final_args.algorithm_params = algorithm_config
        final_args.algorithm = algorithm_config.get('name', 'FedOpt')
        final_args.learning_rate = algorithm_config.get('lr', final_args.learning_rate)
        final_args.local_epochs = algorithm_config.get('local_epochs', final_args.local_epochs)
    else:
        final_args.algorithm = algorithm_config
        final_args.algorithm_params = None

    # Attack
    attack_config = args_dict.get('attack', 'NoAttack')
    if isinstance(attack_config, dict):
        final_args.attack_params = attack_config.get('attack_params')
        final_args.attack = attack_config.get('name', 'NoAttack')
    else:
        final_args.attack = attack_config
        final_args.attack_params = None

    # Defense
    defense_config = args_dict.get('defense', 'Mean')
    if isinstance(defense_config, dict):
        final_args.defense_params = defense_config.get('defense_params')
        final_args.defense = defense_config.get('name', 'Mean')
    else:
        final_args.defense = defense_config
        final_args.defense_params = None

    # Process 'experiment' block to override parameters
    exp_config = args_dict.get('experiment')
    if isinstance(exp_config, dict):
        final_args.epochs = exp_config.get('epochs', final_args.epochs)
        final_args.num_clients = exp_config.get('num_clients', final_args.num_clients)
        final_args.participation_rate = exp_config.get('partial_participation', final_args.participation_rate)
        final_args.clients_per_round = int(final_args.num_clients * final_args.participation_rate) # Recalculate
        final_args.straggler_probability = exp_config.get('straggler_prob', 0.0) # Default if not present
        final_args.straggler_delay = exp_config.get('straggler_delay', 0) # Default if not present
        final_args.verbose = exp_config.get('verbose', final_args.verbose)
        final_args.record_time = exp_config.get('record_time', final_args.record_time)

        attack_exp_config = exp_config.get('attack')
        if isinstance(attack_exp_config, dict):
            final_args.attack_params = attack_exp_config.get('attack_params')
            final_args.attack = attack_exp_config.get('name', 'NoAttack')
        elif attack_exp_config is not None: # If it's a string, override directly
            final_args.attack = attack_exp_config
            final_args.attack_params = None
        
        defense_exp_config = exp_config.get('defense')
        if isinstance(defense_exp_config, dict):
            final_args.defense_params = defense_exp_config.get('defense_params')
            final_args.defense = defense_exp_config.get('name', 'Mean')
        elif defense_exp_config is not None: # If it's a string, override directly
            final_args.defense = defense_exp_config
            final_args.defense_params = None

    # Handle logging block
    logging_config = args_dict.get('logging')
    if isinstance(logging_config, dict):
        final_args.communication_coeff = logging_config.get('communication_coeff', 0.01)
        final_args.energy_coeff = logging_config.get('energy_coeff', 0.01)
        final_args.verbose = logging_config.get('verbose', final_args.verbose)
        final_args.alpha_scaling_factor = logging_config.get('alpha_scaling_factor', 0.1)
        final_args.beta_scaling_factor = logging_config.get('beta_scaling_factor', 1.0)
        final_args.energy_offset = logging_config.get('energy_offset', 10.0)

    # Ensure clients_per_round is calculated if not explicitly set by experiment block
    if not hasattr(final_args, 'clients_per_round'):
        final_args.clients_per_round = max(1, int(final_args.num_clients * final_args.participation_rate))

    # Initialize num_adv based on attack configuration
    # This needs to be done after num_clients is finalized
    if final_args.attack != 'NoAttack':
        num_malicious = args_dict.get('num_malicious')
        if num_malicious is None and final_args.attack_params:
            num_malicious = final_args.attack_params.get('num_malicious')

        if num_malicious is not None:
            final_args.num_adv = frac_or_int_to_int(num_malicious, final_args.num_clients)
        else:
            final_args.num_adv = 0 # Default to 0 if no attack or num_malicious not specified
    else:
        final_args.num_adv = 0 # Default to 0 if no attack

    # Add all_attacks and all_aggregators to args for override_args to use
    final_args.attacks = all_attacks_list
    final_args.defenses = all_aggregators_list
    
    return final_args


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
        if key in ['config', 'attack', 'defense', 'attack_params', 'defense_params', 'benchmark', 'verbose', 'output']:
            continue
        if value is not None:
            setattr(args, key, value)

            print(f"Warning: Overriding {key} with {value}")

    # Handle output specifically
    if cli_args.output is not None:
        setattr(args, 'output', cli_args.output)

    # Handle verbose specifically: only override if cli_args.verbose is explicitly True
    if cli_args.verbose:
        setattr(args, 'verbose', True)

    # override attack, defense, attack_params, defense_params
    for param_type in ['attack', 'defense']:
        cli_arg_value = getattr(cli_args, param_type)
        if cli_arg_value:  # if not None
            setattr(args, param_type, cli_arg_value)
            # if attack_params or defense_params is provided by cli_args, override the corresponding params
            cli_param_value = getattr(cli_args, f"{param_type}_params")
            if cli_param_value:
                # Attempt to parse the string as a dictionary
                try:
                    parsed_params = yaml.safe_load(cli_param_value)
                    if isinstance(parsed_params, dict):
                        setattr(args, f'{param_type}_params', parsed_params)
                    else:
                        print(f"Warning: Could not parse {param_type}_params '{cli_param_value}' as a dictionary. Setting to None.")
                        setattr(args, f'{param_type}_params', None)
                except yaml.YAMLError:
                    print(f"Warning: Could not parse {param_type}_params '{cli_param_value}' as a dictionary. Setting to None.")
                    setattr(args, f'{param_type}_params', None)
            else:
                # if not provided, set the params to default
                # This part needs to be careful as args.attacks and args.defenses are lists of strings
                # and not necessarily dictionaries with 'attack' or 'defense' keys.
                # The original logic here was flawed.
                # If cli_param_value is None, it means no specific params were passed via CLI.
                # The default params should already be set by read_yaml or remain None.
                pass


def benchmark_preprocess(args):
    # This function assumes args.attacks and args.defenses are lists of dictionaries,
    # but they are currently lists of strings. This needs to be addressed.
    # For now, I will assume they are lists of strings and create dummy dicts for iteration.
    
    # Create temporary lists of attack/defense configurations for benchmarking
    # This is a placeholder and might need refinement based on actual benchmark requirements
    temp_attacks = []
    for att_name in args.attacks:
        temp_attacks.append({'attack': att_name, 'attack_params': None})

    temp_defenses = []
    for def_name in args.defenses:
        temp_defenses.append({'defense': def_name, 'defense_params': None})

    for attack_i in temp_attacks:
        for defense_j in temp_defenses:
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
    with open("./configs/dataset_config.yaml", 'r', encoding='utf-8') as file:
        dataset_config = yaml.safe_load(file) # Use safe_load
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

    # Set a default clients_per_round if not already present
    if not hasattr(args, 'clients_per_round'):
        args.clients_per_round = max(1, int(args.num_clients * 0.1)) # Default to 10% of clients
        print(f"Warning: 'clients_per_round' not found in config, setting to default: {args.clients_per_round}")

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

    # If output path is not provided via CLI, generate a default one
    if not hasattr(args, 'output') or args.output is None:
        output_dirname = f'{args.dataset}_{args.model}_{args.distribution}_{args.attack}_{args.defense}_{args.epochs}_{args.num_clients}_{args.learning_rate}_{args.algorithm}'
        args.output = f'./logs/{args.algorithm}/{args.dataset}_{args.model}/{args.distribution}/{output_dirname}'
    
    # Ensure the output directory exists
    os.makedirs(args.output, exist_ok=True)
    return args


def ensure_attr(obj, attr_name):
    """
    set attr_name of obj to None if it does not exist
    """
    if not hasattr(obj, attr_name):
        setattr(obj, attr_name, None)
