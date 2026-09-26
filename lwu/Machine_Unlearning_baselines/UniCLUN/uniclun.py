from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from Machine_Unlearning_baselines.common import (
    clone_frozen,
    features,
    kd_kl,
    paired_batches,
    reset_model,
)


def _contrastive(student, teacher, labels, temperature):
    student = F.normalize(student.flatten(1), dim=1)
    teacher = F.normalize(teacher.flatten(1), dim=1)
    logits = student @ teacher.t() / temperature
    positives = labels[:, None].eq(labels[None, :]).float()
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(positives * log_prob).sum(1).div(positives.sum(1).clamp_min(1)).mean()


class UniCLUN:
    def __init__(
        self,
        model,
        device='cuda',
        buffer_size=500,
        lr=0.01,
        epochs=1,
        rho=2.0,
        contrastive_temperature=0.1,
        alpha1=1.0,
        alpha2=1.0,
        alpha3=1.0,
        momentum=0.99,
        bernoulli_p=0.6,
    ):
        self.model = model
        self.device = device
        self.buffer_size = buffer_size
        self.lr, self.epochs = lr, epochs
        self.rho, self.tau = rho, contrastive_temperature
        self.alpha1, self.alpha2, self.alpha3 = alpha1, alpha2, alpha3
        self.momentum, self.bernoulli_p = momentum, bernoulli_p
        self.buffer_x = None
        self.buffer_y = None
        self.seen = 0

    def _reservoir_add(self, inputs, labels):
        inputs, labels = inputs.detach().cpu(), labels.detach().cpu()
        if self.buffer_x is None:
            self.buffer_x = inputs[: self.buffer_size].clone()
            self.buffer_y = labels[: self.buffer_size].clone()
            self.seen = inputs.size(0)
            return
        for x, y in zip(inputs, labels):
            self.seen += 1
            if self.buffer_x.size(0) < self.buffer_size:
                self.buffer_x = torch.cat((self.buffer_x, x.unsqueeze(0)))
                self.buffer_y = torch.cat((self.buffer_y, y.unsqueeze(0)))
            else:
                index = int(torch.randint(0, self.seen, ()).item())
                if index < self.buffer_size:
                    self.buffer_x[index].copy_(x)
                    self.buffer_y[index].copy_(y)

    def _memory_loader(self, batch_size):
        if self.buffer_x is None or self.buffer_x.size(0) == 0:
            return None
        return DataLoader(
            TensorDataset(self.buffer_x, self.buffer_y), batch_size=batch_size, shuffle=True
        )

    @torch.no_grad()
    def _momentum_update(self, student):
        if torch.rand(()) >= self.bernoulli_p:
            return
        for teacher_parameter, student_parameter in zip(
            self.model.parameters(), student.parameters()
        ):
            teacher_parameter.mul_(self.momentum).add_(student_parameter, alpha=1 - self.momentum)

    def learn(self, task_loader):
        teacher = clone_frozen(self.model)
        student = deepcopy(self.model).to(self.device).train()
        optimizer = torch.optim.SGD(student.parameters(), lr=self.lr, momentum=0.9)
        memory_loader = self._memory_loader(task_loader.batch_size)
        for _ in range(self.epochs):
            if memory_loader is None:
                batches = ((batch, None) for batch in task_loader)
            else:
                batches = paired_batches(task_loader, memory_loader)
            for task_batch, memory_batch in batches:
                task_x, task_y = task_batch
                if memory_batch is None:
                    inputs, labels = task_x, task_y
                    memory_count = 0
                else:
                    memory_x, memory_y = memory_batch
                    inputs = torch.cat((task_x, memory_x))
                    labels = torch.cat((task_y, memory_y))
                    memory_count = memory_x.size(0)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                student_logits = student(inputs)
                loss = F.cross_entropy(student_logits, labels)
                with torch.no_grad():
                    teacher_logits = teacher(inputs)
                    teacher_features = features(teacher, inputs)
                student_features = features(student, inputs)
                if memory_count:
                    sl = slice(-memory_count, None)
                    weights = (
                        F.softmax(teacher_logits[sl] / self.rho, dim=1)
                        .gather(1, labels[sl, None])
                        .squeeze(1)
                    )
                    online = (
                        weights * (teacher_logits[sl] - student_logits[sl]).square().sum(1)
                    ).mean()
                    loss = loss + self.alpha1 * online
                loss = (
                    loss
                    + self.alpha2
                    * _contrastive(student_features, teacher_features, labels, self.tau)
                    + self.alpha3
                    * _contrastive(student_features, student_features, labels, self.tau)
                )
                loss.backward()
                optimizer.step()
                self._momentum_update(student)
        for inputs, labels in task_loader:
            self._reservoir_add(inputs, labels)
        return self.model

    def unlearn(self, forget_loader, retain_loader=None):
        student = deepcopy(self.model).to(self.device).train()
        good_teacher = clone_frozen(self.model)
        bad_teacher = reset_model(deepcopy(self.model)).to(self.device).eval()
        for parameter in bad_teacher.parameters():
            parameter.requires_grad_(False)
        if retain_loader is None:
            retain_loader = self._memory_loader(forget_loader.batch_size)
        if retain_loader is None:
            raise RuntimeError('UniCLUN unlearning requires its bounded replay buffer')
        optimizer = torch.optim.SGD(student.parameters(), lr=self.lr, momentum=0.9)
        forgotten_classes = set()
        for _ in range(self.epochs):
            for (forget_x, forget_y), (retain_x, _) in paired_batches(forget_loader, retain_loader):
                forgotten_classes.update(int(v) for v in forget_y.unique())
                forget_x, retain_x = forget_x.to(self.device), retain_x.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    bad_logits = bad_teacher(forget_x)
                    good_logits = good_teacher(retain_x)
                loss = kd_kl(student(forget_x), bad_logits) + kd_kl(student(retain_x), good_logits)
                loss.backward()
                optimizer.step()
                self._momentum_update(student)
        if self.buffer_y is not None:
            keep = torch.ones_like(self.buffer_y, dtype=torch.bool)
            for class_id in forgotten_classes:
                keep &= self.buffer_y.ne(class_id)
            self.buffer_x, self.buffer_y = self.buffer_x[keep], self.buffer_y[keep]
        self.model.load_state_dict(student.state_dict())
        return self.model
