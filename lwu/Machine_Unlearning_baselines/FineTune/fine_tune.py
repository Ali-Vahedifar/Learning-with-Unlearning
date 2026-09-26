from Machine_Unlearning_baselines.common import train_classifier


class FineTune:
    def __init__(self, model, device='cuda', epochs=5, lr=1e-3, optimizer='adam'):
        self.model = model
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.optimizer = optimizer

    def unlearn(self, forget_loader, retain_loader):
        del forget_loader
        return train_classifier(
            self.model, retain_loader, self.device, self.epochs, self.lr, self.optimizer
        )
