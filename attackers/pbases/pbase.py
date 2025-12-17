class PBase():
    def __init__(self, *args, **kwargs):
        self.args = kwargs.get('args')
        self.worker_id = kwargs.get('worker_id')
        self.default_attack_params = {}
        self.update_and_set_attr()
        super().__init__()

    def update_and_set_attr(self):
        """
        update the attack parameters if given, otherwise use the default attack parameters
        """
        new_attack_params = getattr(self.args, "attack_params", None)
        self.attack_params = self.default_attack_params.copy() # Use a copy to avoid modifying the default

        # update default attack params with new attack_params
        if new_attack_params:
            self.attack_params.update(
                new_attack_params)
        # set the attack parameters as the class attributes
        for key, value in self.attack_params.items():
            setattr(self, key, value)
