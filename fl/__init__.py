import os
import importlib
import glob

class Register(dict):
    def __call__(self, cls):
        self[cls.__name__] = cls
        return cls

aggregator_registry = Register()

def import_all_modules(directory):
    py_files = glob.glob(os.path.join(directory, "*.py"))
    for file in py_files:
        module_name = os.path.basename(file)[:-3]
        if module_name != "__init__":
            importlib.import_module(f"{module_name}")

def get_aggregator(name):
    if name in aggregator_registry:
        return aggregator_registry[name]
    raise KeyError(f"Aggregator {name} not found")

all_aggregators = list(aggregator_registry.keys())
