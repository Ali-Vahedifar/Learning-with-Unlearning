import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from Machine_Unlearning_baselines.common import train_classifier


class UNSIR:
    def __init__(
        self,
        model,
        device='cuda',
        noise_steps=40,
        noise_lr=0.1,
        noise_regularization=0.1,
        impair_lr=0.01,
        repair_lr=0.01,
        impair_epochs=1,
        repair_epochs=1,
    ):
        self.model = model
        self.device = device
        self.noise_steps = noise_steps
        self.noise_lr = noise_lr
        self.noise_regularization = noise_regularization
        self.impair_lr = impair_lr
        self.repair_lr = repair_lr
        self.impair_epochs = impair_epochs
        self.repair_epochs = repair_epochs

    def _forget_classes_and_shape(self, forget_loader):
        classes = set()
        shape = None
        for inputs, targets in forget_loader:
            shape = tuple(inputs.shape[1:])
            classes.update(int(v) for v in targets.unique())
        if shape is None:
            raise ValueError('forget loader is empty')
        return sorted(classes), shape

    def generate_error_maximizing_noise(self, forget_loader):
        classes, input_shape = self._forget_classes_and_shape(forget_loader)
        was_training = self.model.training
        self.model.eval()
        noises, labels = [], []
        for class_id in classes:
            noise = torch.randn((1,) + input_shape, device=self.device, requires_grad=True)
            target = torch.tensor([class_id], device=self.device)
            optimizer = torch.optim.Adam([noise], lr=self.noise_lr)
            for _ in range(self.noise_steps):
                optimizer.zero_grad(set_to_none=True)
                objective = (
                    -F.cross_entropy(self.model(noise), target)
                    + self.noise_regularization * noise.square().mean()
                )
                objective.backward()
                optimizer.step()
            noises.append(noise.detach().cpu())
            labels.append(class_id)
        self.model.train(was_training)
        return torch.cat(noises), torch.tensor(labels, dtype=torch.long)

    def unlearn(self, forget_loader, retain_loader):
        noise, labels = self.generate_error_maximizing_noise(forget_loader)
        retain_x, retain_y = [], []
        for inputs, targets in retain_loader:
            retain_x.append(inputs.cpu())
            retain_y.append(targets.cpu())
        if not retain_x:
            raise ValueError('retain loader is empty')
        repeat = max(1, sum(x.size(0) for x in retain_x) // max(1, len(labels)))
        impair_x = torch.cat(retain_x + [noise.repeat_interleave(repeat, dim=0)])
        impair_y = torch.cat(retain_y + [labels.repeat_interleave(repeat)])
        impair_loader = DataLoader(
            TensorDataset(impair_x, impair_y), batch_size=retain_loader.batch_size, shuffle=True
        )
        train_classifier(
            self.model, impair_loader, self.device, self.impair_epochs, self.impair_lr, 'sgd'
        )
        train_classifier(
            self.model, retain_loader, self.device, self.repair_epochs, self.repair_lr, 'sgd'
        )
        return self.model
