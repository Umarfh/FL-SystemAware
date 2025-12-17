import os
from global_utils import import_all_modules, Register

aggregator_registry = Register()

# Automatically import all aggregator modules in this directory
current_dir = os.path.dirname(__file__)
import_all_modules(current_dir)

# List all registered aggregators
all_aggregators = list(aggregator_registry.keys())
print(f"[DEBUG] Available aggregators: {all_aggregators}")

def get_aggregator(name):
    """Return the aggregator class by name."""
    return aggregator_registry[name]
