from copy import deepcopy

import torch

from Machine_Unlearning_baselines.common import clone_frozen, kd_kl, paired_batches, reset_model


class BadTeacher:
    def __init__(self, model, device='cuda', temperature=1.0, epochs=5, lr=0.01):
        self.model = model
        self.device = device
        self.temperature = temperature
        self.epochs = epochs
        self.lr = lr

    def unlearn(self, forget_loader, retain_loader):
        competent = clone_frozen(self.model)
        incompetent = reset_model(deepcopy(self.model)).to(self.device).eval()
        for parameter in incompetent.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.model.train()
        for _ in range(self.epochs):
            for (forget_x, _), (retain_x, _) in paired_batches(forget_loader, retain_loader):
                forget_x, retain_x = forget_x.to(self.device), retain_x.to(self.device)
                inputs = torch.cat((forget_x, retain_x), dim=0)
                with torch.no_grad():
                    targets = torch.cat((incompetent(forget_x), competent(retain_x)), dim=0)
                optimizer.zero_grad(set_to_none=True)
                loss = kd_kl(self.model(inputs), targets, self.temperature)
                loss.backward()
                optimizer.step()
        return self.model
