from Machine_Unlearning_baselines.common import train_classifier


class Retrain:
    def __init__(self, model_factory, device='cuda', epochs=100, lr=1e-3, optimizer='adam'):
        self.model_factory = model_factory
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optimizer
        self.model = None

    def unlearn(self, forget_loader, retain_loader):
        del forget_loader
        self.model = self.model_factory().to(self.device)
        return train_classifier(
            self.model, retain_loader, self.device, self.epochs, self.lr, self.optimizer
        )
