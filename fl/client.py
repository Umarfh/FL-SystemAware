import torch
import random
import torch.nn as nn # Added for criterion
import torch.optim as optim # Added for optimized client training
import copy # Added for deepcopy
from fl.algorithms import get_algorithm_handler
from fl.models import get_model
from fl.models.model_utils import vec2model
from fl.worker import Worker
from global_utils import actor, avg_value, TimingRecorder


@actor("benign", "always")
class Client(Worker):
    def __init__(self, *args, **kwargs):
        # Extract arguments needed by Client
        args_val = kwargs.pop('args')
        worker_id_val = kwargs.pop('worker_id')
        train_dataset_val = kwargs.pop('train_dataset')
        test_dataset_val = kwargs.pop('test_dataset', None)

        # Pass remaining args and kwargs to super()
        super().__init__(*args, args=args_val, worker_id=worker_id_val, **kwargs)

        # Explicitly set self.args to ensure it's available
        self.args = args_val

        self.client_id = worker_id_val
        self.train_dataset = train_dataset_val
        self.test_dataset = test_dataset_val
        self.global_epoch = 0
        self.client_test_accuracy = 0.0 # Added for client-level accuracy tracking

        # Initialize model
        self.model = get_model(args_val)
        print(f"DEBUG: Client {self.client_id} model type: {type(self.model)}")
        print(f"DEBUG: Client {self.client_id} model total parameters: {sum(p.numel() for p in self.model.parameters())}")
        self.global_weights_vec = None # Initialize global_weights_vec to None, will be set by load_global_model

        # Remove optimizer and scheduler initialization from __init__
        # These will be handled within the new set_algorithm method
        self.optimizer = None
        self.lr_scheduler = None

        # -------------------------
        # Ensure a default loss is set for classification
        if not hasattr(self, 'criterion_fn') or self.criterion_fn is None:
            # Use cross-entropy for classification by default
            self.criterion_fn = torch.nn.CrossEntropyLoss()
            self.logger = getattr(self, 'logger', None)
            if self.logger:
                self.logger.info(f"[Client {self.client_id}] criterion set to CrossEntropyLoss")
        # -------------------------

        print(f"[Client {self.client_id}] args.algorithm: {args_val.algorithm}")
        print(f"[Client {self.client_id}] args.local_epochs: {args_val.local_epochs}")

        self.train_loader = self.get_dataloader(self.train_dataset, train_flag=True)
        self.record_time(getattr(args_val, "record_time", False))
        self.set_algorithm(args_val.algorithm) # Initialize the algorithm and local_epochs
        print(f"[Client {self.client_id}] self.local_epochs after set_algorithm: {self.local_epochs}")

    def record_time(self, record_time):
        if record_time:
            self.time_recorder = TimingRecorder(self.worker_id, self.args.output)
            self.local_training = self.time_recorder.timing_decorator(self.local_training)
            self.fetch_updates = self.time_recorder.timing_decorator(self.fetch_updates)

    def set_algorithm(self, algorithm):
        """
        Fixed version that properly reads from config
        """
        self.algorithm = get_algorithm_handler(algorithm)(self.args, self.model, self.optimizer) # Still need to pass optimizer, even if None initially

        # CRITICAL: Read local_epochs from args, not hardcoded
        if hasattr(self.args, 'local_epochs'):
            self.local_epochs = self.args.local_epochs
        else:
            self.local_epochs = 10  # Default to 10, not 3!

        # Debug print to verify
        print(f"[Client {self.client_id}] Set local_epochs to: {self.local_epochs}")

        # Set optimizer with correct learning rate
        if hasattr(self.args, 'lr'):
            lr = self.args.lr
        else:
            lr = 0.1  # Default to 0.1 for SGD, not 0.01!

        # Re-initialize optimizer based on algorithm and args
        if algorithm.lower() in ["fedavg", "fedprox", "fedopt"]: # Use .lower() for case-insensitivity
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=getattr(self.args, 'momentum', 0.9),
                weight_decay=getattr(self.args, 'weight_decay', 0.0005),
                nesterov=getattr(self.args, 'nesterov', True)
            )
            print(f"[Client {self.client_id}] Optimizer: SGD with lr={lr}, momentum={getattr(self.args, 'momentum', 0.9)}")
        else:
            # Fallback for other algorithms if needed, or raise error
            print(f"[Client {self.client_id}] Warning: Optimizer not explicitly set for algorithm {algorithm}. Using default SGD.")
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        # Initialize LR scheduler
        if getattr(self.args, 'use_scheduler', False) and getattr(self.args, 'scheduler', None) == 'step':
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=getattr(self.args, 'step_size', self.local_epochs), gamma=getattr(self.args, 'gamma', 0.1))
        else:
            self.lr_scheduler = None # No scheduler if not configured

        # The original line `self.local_epochs = self.algorithm.init_local_epochs()` is now effectively overridden
        # by the explicit setting above, which is what the diagnostic suggested.

    def load_global_model(self, global_weights_vec):
        self.global_weights_vec = global_weights_vec
        vec2model(self.global_weights_vec, self.model)

    def local_training(self, model=None, train_loader=None, optimizer=None, criterion_fn=None, local_epochs=None):
        """
        Robust local training loop (replace or patch existing implementation).
        Returns: (avg_acc, avg_loss)
        """
        model = self.new_if_given(model, self.model)
        train_loader = self.new_if_given(train_loader, self.train_loader)
        optimizer = self.new_if_given(optimizer, self.optimizer)
        criterion_fn = self.new_if_given(criterion_fn, self.criterion_fn)
        local_epochs = self.new_if_given(local_epochs, self.local_epochs)

        model.train()
        epoch_accs = []
        epoch_losses = []

        # If train_loader is an iterator generator (some clients do that), convert to a fresh generator each epoch:
        is_generator = not isinstance(train_loader, torch.utils.data.DataLoader)

        for epoch in range(local_epochs):
            print(f"DEBUG: Client {self.client_id} - Epoch {epoch+1}/{local_epochs} starting...")
            running_loss = 0.0
            running_correct = 0
            running_total = 0

            loader = train_loader() if is_generator and callable(train_loader) else train_loader
            # print(f"DEBUG: Client {self.client_id} - DataLoader created for epoch {epoch+1}. Length: {len(loader.dataset) if hasattr(loader, 'dataset') else 'N/A'}")

            for i, batch in enumerate(loader):
                # print(f"DEBUG: Client {self.client_id} - Epoch {epoch+1}, Batch {i+1} starting...")
                # batch can be (inputs, targets) or more
                inputs, targets = batch[0], batch[1]
                inputs = inputs.to(self.args.device)
                targets = targets.to(self.args.device)

                optimizer.zero_grad()
                outputs = model(inputs)

                # ensure outputs shape matches expected
                loss = criterion_fn(outputs, targets)
                loss.backward()
                optimizer.step()

                # stats
                running_loss += loss.item() * targets.size(0)
                preds = outputs.argmax(dim=1)
                running_correct += (preds == targets).sum().item()
                running_total += targets.size(0)

            # end epoch
            epoch_loss = running_loss / max(1, running_total)
            epoch_acc = running_correct / max(1, running_total)
            epoch_losses.append(epoch_loss)
            epoch_accs.append(epoch_acc)

            # optional lr schedule step if present
            try:
                if hasattr(self.lr_scheduler, 'step'):
                    self.lr_scheduler.step()
            except Exception:
                pass

        self.avg_local_acc = sum(epoch_accs) / len(epoch_accs) if epoch_accs else 0.0
        self.avg_local_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0

        # Perform client-level test after local training
        if self.test_dataset:
            self.client_test_accuracy, _ = self.client_test(model, self.test_dataset)

        return self.avg_local_acc, self.avg_local_loss

    def fetch_updates(self, benign_flag=False):
        # Perform local training first
        self.local_training(model=self.model, train_loader=self.train_loader, 
                            optimizer=self.optimizer, criterion_fn=self.criterion_fn, 
                            local_epochs=self.local_epochs)
        
        # Then get the update based on the trained model
        self.update = self.algorithm.get_local_update(global_weights_vec=self.global_weights_vec)
        self.global_epoch += 1

    def client_test(self, model=None, test_dataset=None):
        model = self.new_if_given(model, self.model)
        test_dataset = self.new_if_given(test_dataset, self.test_dataset)
        test_loader = self.get_dataloader(test_dataset, train_flag=False)
        test_acc, test_loss = self.test(model, test_loader)
        return test_acc, test_loss

    @staticmethod
    def select_clients(clients, participation_rate=0.6):
        num_selected = max(1, int(len(clients) * participation_rate))
        return random.sample(clients, num_selected)
