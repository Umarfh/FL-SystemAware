import numpy as np
from global_utils import actor
from attackers.pbases.mpbase import MPBase
from attackers import attacker_registry
from fl.client import Client


@attacker_registry.register("Mimic", ["model_poisoning", "omniscient"], "attacker")
@actor('attacker', 'model_poisoning', 'omniscient')
class Mimic(MPBase, Client):
    """
    [Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing](https://openreview.net/forum?id=jXKKDEi5vJt) - ICLR '22
    Mimic the behavior of any fixed benign client to introduce consistent bias that overemphasizes the influence of that benign update while under-representing other benign updates.
    """

    def __init__(self, *args, choice=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_attack_params = {'choice': choice}
        self.update_and_set_attr()
        self.algorithm = "FedSGD"

    def omniscient(self, clients):
        assert self.choice < len(
            clients), f"choice {self.choice} is out of range"
        attack_vec = clients[self.choice].update
        # repeat attack vector for all attackers
        return np.tile(attack_vec, (self.args.num_adv, 1))
