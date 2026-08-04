"""
Baseline Methods for Continual Learning

Implements:
- Elastic Weight Consolidation (EWC)
- Synaptic Intelligence (SI)
- Learning without Forgetting (LwF)
- SGD (lower bound)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
import numpy as np


class EWC:
    """
    Elastic Weight Consolidation (EWC)
    
    Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (2017)
    
    Adds a quadratic penalty to protect important parameters based on
    Fisher Information.
    """
    
    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 100.0,
        gamma: float = 1.0,
        device: str = 'cuda'
    ):
        """
        Initialize EWC.
        
        Args:
            model: Neural network model
            ewc_lambda: Regularization strength
            gamma: Decay factor for older tasks
            device: Device to use
        """
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.gamma = gamma
        self.device = device
        
        # Store Fisher information and parameters for each task
        self.fisher_matrices = []
        self.optimal_params = []
        
    def compute_fisher(
        self,
        dataloader: DataLoader,
        num_samples: int = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information for current task.
        
        Args:
            dataloader: DataLoader for current task data
            num_samples: Number of samples to use (None = all)
            
        Returns:
            Dictionary mapping parameter names to Fisher values
        """
        fisher = {name: torch.zeros_like(param) 
                  for name, param in self.model.named_parameters() 
                  if param.requires_grad}
        
        self.model.eval()
        count = 0
        
        for inputs, targets in dataloader:
            if num_samples and count >= num_samples:
                break
                
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            
            # Use log-likelihood
            log_probs = F.log_softmax(outputs, dim=1)
            loss = F.nll_loss(log_probs, targets)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2
                    
            count += inputs.size(0)
            
        # Normalize
        for name in fisher:
            fisher[name] /= count
            
        self.model.train()
        return fisher
    
    def consolidate(self, dataloader: DataLoader):
        """
        Consolidate knowledge after training on a task.
        
        Args:
            dataloader: DataLoader for the task just completed
        """
        # Compute Fisher Information
        fisher = self.compute_fisher(dataloader)
        self.fisher_matrices.append(fisher)
        
        # Store optimal parameters
        optimal_params = {name: param.data.clone() 
                        for name, param in self.model.named_parameters()
                        if param.requires_grad}
        self.optimal_params.append(optimal_params)
        
        # Apply decay to older tasks
        for i in range(len(self.fisher_matrices) - 1):
            for name in self.fisher_matrices[i]:
                self.fisher_matrices[i][name] *= self.gamma
                
    def penalty(self) -> torch.Tensor:
        """
        Compute EWC penalty term.
        
        Returns:
            EWC regularization loss
        """
        if not self.fisher_matrices:
            return torch.tensor(0.0, device=self.device)
            
        loss = torch.tensor(0.0, device=self.device)
        
        for task_idx, (fisher, optimal) in enumerate(
            zip(self.fisher_matrices, self.optimal_params)
        ):
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in fisher:
                    diff = param - optimal[name]
                    loss += (fisher[name] * diff ** 2).sum()
                    
        return self.ewc_lambda * loss


class SI:
    """
    Synaptic Intelligence (SI)
    
    Zenke et al., "Continual Learning Through Synaptic Intelligence" (2017)
    
    Online computation of parameter importance based on contribution to
    loss changes during training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        si_c: float = 1.0,
        xi: float = 1.0,
        device: str = 'cuda'
    ):
        """
        Initialize SI.
        
        Args:
            model: Neural network model
            si_c: Regularization strength
            xi: Damping factor
            device: Device to use
        """
        self.model = model
        self.si_c = si_c
        self.xi = xi
        self.device = device
        
        # Initialize tracking variables
        self.omega = {}  # Importance weights
        self.prev_params = {}  # Parameters at task start
        self.W = {}  # Path integral (accumulated gradients)
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.omega[name] = torch.zeros_like(param)
                self.prev_params[name] = param.data.clone()
                self.W[name] = torch.zeros_like(param)
                
    def update_omega(self):
        """Update omega values after completing a task."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                delta = param.data - self.prev_params[name]
                # Compute omega contribution
                omega_add = self.W[name] / (delta ** 2 + self.xi)
                self.omega[name] += omega_add.clamp(min=0)
                
                # Reset for next task
                self.prev_params[name] = param.data.clone()
                self.W[name].zero_()
                
    def accumulate_grad(self):
        """Accumulate gradient information during training."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # W tracks the path integral: W += -grad * delta
                delta = param.data - self.prev_params[name]
                self.W[name] += (-param.grad.data * delta)
                
    def penalty(self) -> torch.Tensor:
        """
        Compute SI penalty term.
        
        Returns:
            SI regularization loss
        """
        loss = torch.tensor(0.0, device=self.device)
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.omega:
                diff = param - self.prev_params[name]
                loss += (self.omega[name] * diff ** 2).sum()
                
        return self.si_c * loss


class LwF:
    """
    Learning without Forgetting (LwF)
    
    Li & Hoiem, "Learning without Forgetting" (2017)
    
    Uses knowledge distillation to preserve old task knowledge.
    """
    
    def __init__(
        self,
        model: nn.Module,
        temperature: float = 2.0,
        alpha: float = 0.5,
        device: str = 'cuda'
    ):
        """
        Initialize LwF.
        
        Args:
            model: Neural network model
            temperature: Distillation temperature
            alpha: Weight for distillation loss
            device: Device to use
        """
        self.model = model
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        
        # Stored teacher model (snapshot before new task)
        self.teacher_model = None
        self.old_classes = 0
        
    def store_teacher(self, num_classes: int):
        """
        Store current model as teacher.
        
        Args:
            num_classes: Number of classes seen so far
        """
        self.teacher_model = deepcopy(self.model)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.old_classes = num_classes
        
    def distillation_loss(
        self,
        inputs: torch.Tensor,
        student_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distillation loss.
        
        Args:
            inputs: Input data
            student_outputs: Student model outputs
            
        Returns:
            Distillation loss
        """
        if self.teacher_model is None or self.old_classes == 0:
            return torch.tensor(0.0, device=self.device)
            
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)
            
        # Only use outputs for old classes
        teacher_probs = F.softmax(teacher_outputs[:, :self.old_classes] / self.temperature, dim=1)
        student_probs = F.log_softmax(student_outputs[:, :self.old_classes] / self.temperature, dim=1)
        
        loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
        loss = loss * (self.temperature ** 2)
        
        return self.alpha * loss
    
    def combined_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        outputs: torch.Tensor,
        criterion: nn.Module
    ) -> torch.Tensor:
        """
        Compute combined classification and distillation loss.
        
        Args:
            inputs: Input data
            targets: Ground truth labels
            outputs: Model outputs
            criterion: Classification loss function
            
        Returns:
            Combined loss
        """
        # Classification loss
        cls_loss = criterion(outputs, targets)
        
        # Distillation loss
        dist_loss = self.distillation_loss(inputs, outputs)
        
        return cls_loss + dist_loss


class SGDBaseline:
    """
    SGD Baseline (lower bound)
    
    Standard SGD training without any continual learning mechanism.
    Serves as a lower bound for comparison.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    ):
        """
        Initialize SGD baseline.
        
        Args:
            model: Neural network model
            device: Device to use
        """
        self.model = model
        self.device = device
        
    def penalty(self) -> torch.Tensor:
        """No penalty for SGD baseline."""
        return torch.tensor(0.0, device=self.device)


class JointTraining:
    """
    Joint Training (upper bound)
    
    Trains on all data simultaneously.
    Serves as an upper bound for comparison.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    ):
        """
        Initialize joint training.
        
        Args:
            model: Neural network model
            device: Device to use
        """
        self.model = model
        self.device = device
        self.all_data = []
        
    def add_task_data(self, dataloader: DataLoader):
        """Store data for joint training."""
        for inputs, targets in dataloader:
            self.all_data.append((inputs, targets))
            
    def get_joint_loader(self, batch_size: int = 64) -> DataLoader:
        """Create a loader with all accumulated data."""
        from torch.utils.data import TensorDataset
        
        all_inputs = torch.cat([d[0] for d in self.all_data], dim=0)
        all_targets = torch.cat([d[1] for d in self.all_data], dim=0)
        
        dataset = TensorDataset(all_inputs, all_targets)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_continual_method(
    name: str,
    model: nn.Module,
    device: str = 'cuda',
    **kwargs
):
    """
    Factory function to create continual learning method.
    
    Args:
        name: Method name ('ewc', 'si', 'lwf', 'sgd', 'joint')
        model: Neural network model
        device: Device to use
        **kwargs: Method-specific arguments
        
    Returns:
        Continual learning method instance
    """
    methods = {
        'ewc': lambda: EWC(model, device=device, **kwargs),
        'si': lambda: SI(model, device=device, **kwargs),
        'lwf': lambda: LwF(model, device=device, **kwargs),
        'sgd': lambda: SGDBaseline(model, device=device),
        'joint': lambda: JointTraining(model, device=device)
    }
    
    if name.lower() not in methods:
        raise ValueError(f"Unknown method: {name}. Available: {list(methods.keys())}")
        
    return methods[name.lower()]()
