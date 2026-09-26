class Baseline:
    def __init__(self, model, **_):
        self.model = model

    def unlearn(self, *_args, **_kwargs):
        return self.model
