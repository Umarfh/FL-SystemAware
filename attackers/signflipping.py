from fl.client import Client
from global_utils import actor
from attackers.pbases.mpbase import MPBase
from attackers import attacker_registry


@attacker_registry.register("SignFlipping", ["model_poisoning", "non_omniscient"], "attacker")
@actor('attacker', 'model_poisoning', 'non_omniscient')
class SignFlipping(MPBase, Client):
    """
    [Asynchronous Byzantine machine learning (the case of SGD)](http://proceedings.mlr.press/v80/damaskinos18a/damaskinos18a.pdf) - ICML '18
    reverse the sign of the update
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def non_omniscient(self):
        return - self.update
