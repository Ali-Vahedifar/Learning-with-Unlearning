"""
Baseline Methods for Machine Unlearning

Implements:
- Selective Synaptic Dampening (SSD)
- Bad Teacher
- Amnesiac
- UNSIR (Unlearning by Selective Impair and Repair)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
import numpy as np


class SSD:
    """
    Selective Synaptic Dampening (SSD)
    
    Foster et al., "Fast Machine Unlearning Without Retraining Through 
    Selective Synaptic Dampening" (2024)
    
    Identifies and dampens synapses associated with forget data.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        dampening_constant: float = 1.0,
        selection_weighting: float = 10.0
    ):
        """
        Initialize SSD.
        
        Args:
            model: Neural network model
            device: Device to use
            dampening_constant: Controls dampening strength
            selection_weighting: Controls selection of parameters to dampen
        """
        self.model = model
        self.device = device
        self.dampening_constant = dampening_constant
        self.selection_weighting = selection_weighting
        
    def compute_fisher(
        self,
        dataloader: DataLoader
    ) -> Dict[str, torch.Tensor]:
        """Compute Fisher Information for parameters."""
        fisher = {name: torch.zeros_like(param) 
                  for name, param in self.model.named_parameters() 
                  if param.requires_grad}
        
        self.model.eval()
        count = 0
        
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2
                    
            count += inputs.size(0)
            
        for name in fisher:
            fisher[name] /= count
            
        self.model.train()
        return fisher
    
    def unlearn(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader
    ):
        """
        Perform SSD unlearning.
        
        Args:
            forget_loader: DataLoader for forget set
            retain_loader: DataLoader for retain set
        """
        # Compute Fisher for forget and retain sets
        fisher_forget = self.compute_fisher(forget_loader)
        fisher_retain = self.compute_fisher(retain_loader)
        
        # Compute selection mask
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Parameters with high forget importance and low retain importance
                f_f = fisher_forget[name]
                f_r = fisher_retain[name]
                
                # Selection based on relative importance
                selection = f_f / (f_r + 1e-8)
                selection = selection / (selection.max() + 1e-8)
                
                # Apply dampening
                dampening = 1.0 - self.dampening_constant * (selection ** self.selection_weighting)
                dampening = dampening.clamp(min=0.0)
                
                param.data = param.data * dampening


class BadTeacher:
    """
    Bad Teacher
    
    Chundawat et al., "Can Bad Teaching Induce Forgetting?" (2023)
    
    Uses a randomly initialized "bad teacher" to corrupt forget knowledge.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        kd_temperature: float = 1.0,
        num_epochs: int = 5,
        lr: float = 0.01
    ):
        """
        Initialize Bad Teacher.
        
        Args:
            model: Neural network model
            device: Device to use
            kd_temperature: Knowledge distillation temperature
            num_epochs: Number of unlearning epochs
            lr: Learning rate for unlearning
        """
        self.model = model
        self.device = device
        self.kd_temperature = kd_temperature
        self.num_epochs = num_epochs
        self.lr = lr
        
    def unlearn(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader
    ):
        """
        Perform Bad Teacher unlearning.
        
        Args:
            forget_loader: DataLoader for forget set
            retain_loader: DataLoader for retain set
        """
        # Create bad teacher (randomly initialized)
        bad_teacher = deepcopy(self.model)
        for param in bad_teacher.parameters():
            if len(param.shape) >= 2:
                nn.init.kaiming_normal_(param)
            else:
                nn.init.zeros_(param)
        bad_teacher.eval()
        
        # Create good teacher (current model)
        good_teacher = deepcopy(self.model)
        good_teacher.eval()
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        
        self.model.train()
        
        for epoch in range(self.num_epochs):
            # Unlearn on forget set using bad teacher
            for inputs, _ in forget_loader:
                inputs = inputs.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(inputs)
                
                with torch.no_grad():
                    bad_outputs = bad_teacher(inputs)
                    
                # KD loss towards bad teacher
                loss = F.kl_div(
                    F.log_softmax(outputs / self.kd_temperature, dim=1),
                    F.softmax(bad_outputs / self.kd_temperature, dim=1),
                    reduction='batchmean'
                ) * (self.kd_temperature ** 2)
                
                loss.backward()
                optimizer.step()
                
            # Repair on retain set using good teacher
            for inputs, targets in retain_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(inputs)
                
                # Classification loss
                cls_loss = F.cross_entropy(outputs, targets)
                
                # KD loss towards good teacher
                with torch.no_grad():
                    good_outputs = good_teacher(inputs)
                    
                kd_loss = F.kl_div(
                    F.log_softmax(outputs / self.kd_temperature, dim=1),
                    F.softmax(good_outputs / self.kd_temperature, dim=1),
                    reduction='batchmean'
                ) * (self.kd_temperature ** 2)
                
                loss = cls_loss + 0.5 * kd_loss
                loss.backward()
                optimizer.step()


class Amnesiac:
    """
    Amnesiac Unlearning
    
    Graves et al., "Amnesiac Machine Learning" (2021)
    
    Stores gradients during training and subtracts them for unlearning.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        unlearning_rate: float = 1.0
    ):
        """
        Initialize Amnesiac.
        
        Args:
            model: Neural network model
            device: Device to use
            unlearning_rate: Multiplier for gradient subtraction
        """
        self.model = model
        self.device = device
        self.unlearning_rate = unlearning_rate
        
        # Store cumulative gradients during training
        self.stored_gradients = {}
        
    def store_gradients(self, dataloader: DataLoader, num_epochs: int = 1):
        """
        Store gradients for potential unlearning.
        
        Should be called during original training.
        
        Args:
            dataloader: DataLoader for data to potentially unlearn
            num_epochs: Number of training epochs
        """
        criterion = nn.CrossEntropyLoss()
        
        # Initialize storage
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.stored_gradients[name] = torch.zeros_like(param)
                
        self.model.eval()
        
        for _ in range(num_epochs):
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                self.model.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self.stored_gradients[name] += param.grad.data
                        
        self.model.train()
        
    def unlearn(
        self,
        forget_loader: DataLoader = None,
        retain_loader: DataLoader = None
    ):
        """
        Perform Amnesiac unlearning by subtracting stored gradients.
        
        Args:
            forget_loader: Not used (gradients should be pre-stored)
            retain_loader: Not used
        """
        if not self.stored_gradients:
            raise ValueError("No gradients stored. Call store_gradients first.")
            
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.stored_gradients:
                param.data -= self.unlearning_rate * self.stored_gradients[name]


class UNSIR:
    """
    UNSIR (Unlearning by Selective Impair and Repair)
    
    Tarun et al., "Fast Yet Effective Machine Unlearning" (2023)
    
    Two-phase approach: impair on forget set, repair on retain set.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        impair_lr: float = 0.1,
        repair_lr: float = 0.01,
        impair_epochs: int = 3,
        repair_epochs: int = 5
    ):
        """
        Initialize UNSIR.
        
        Args:
            model: Neural network model
            device: Device to use
            impair_lr: Learning rate for impair phase
            repair_lr: Learning rate for repair phase
            impair_epochs: Number of impair epochs
            repair_epochs: Number of repair epochs
        """
        self.model = model
        self.device = device
        self.impair_lr = impair_lr
        self.repair_lr = repair_lr
        self.impair_epochs = impair_epochs
        self.repair_epochs = repair_epochs
        
    def _impair(self, forget_loader: DataLoader):
        """Impair phase: maximize loss on forget set."""
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.impair_lr,
            momentum=0.9
        )
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        
        for epoch in range(self.impair_epochs):
            for inputs, targets in forget_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                
                # Negative loss to maximize error on forget set
                loss = -criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
    def _repair(self, retain_loader: DataLoader):
        """Repair phase: fine-tune on retain set."""
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.repair_lr,
            momentum=0.9,
            weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        
        for epoch in range(self.repair_epochs):
            for inputs, targets in retain_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
    def unlearn(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader
    ):
        """
        Perform UNSIR unlearning.
        
        Args:
            forget_loader: DataLoader for forget set
            retain_loader: DataLoader for retain set
        """
        # Phase 1: Impair
        self._impair(forget_loader)
        
        # Phase 2: Repair
        self._repair(retain_loader)


def retrain_from_scratch(
    model_class,
    retain_loader: DataLoader,
    num_classes: int,
    device: str = 'cuda',
    num_epochs: int = 100,
    lr: float = 0.001,
    **model_kwargs
) -> nn.Module:
    """
    Retrain model from scratch (gold standard for unlearning).
    
    Args:
        model_class: Model class to instantiate
        retain_loader: DataLoader for retain set
        num_classes: Number of output classes
        device: Device to use
        num_epochs: Number of training epochs
        lr: Learning rate
        **model_kwargs: Additional model arguments
        
    Returns:
        Retrained model
    """
    model = model_class(num_classes=num_classes, **model_kwargs).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    for epoch in range(num_epochs):
        for inputs, targets in retain_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
    return model


def get_unlearning_method(
    name: str,
    model: nn.Module,
    device: str = 'cuda',
    **kwargs
):
    """
    Factory function to create unlearning method.
    
    Args:
        name: Method name ('ssd', 'bad_teacher', 'amnesiac', 'unsir')
        model: Neural network model
        device: Device to use
        **kwargs: Method-specific arguments
        
    Returns:
        Unlearning method instance
    """
    methods = {
        'ssd': lambda: SSD(model, device=device, **kwargs),
        'bad_teacher': lambda: BadTeacher(model, device=device, **kwargs),
        'amnesiac': lambda: Amnesiac(model, device=device, **kwargs),
        'unsir': lambda: UNSIR(model, device=device, **kwargs)
    }
    
    if name.lower() not in methods:
        raise ValueError(f"Unknown method: {name}. Available: {list(methods.keys())}")
        
    return methods[name.lower()]()
