"""
Learning with Unlearning (LwU) - Main Model Implementation

A unified framework for simultaneous Continual Learning and Machine Unlearning
through principled parameter space decomposition into four topological zones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from copy import deepcopy


class LwU(nn.Module):
    """
    Learning with Unlearning (LwU) Framework
    
    Decomposes neural network parameters into four zones:
    - Zone A (Safe Retain): Parameters exclusive to retained tasks
    - Zone B (Pure Forget): Parameters exclusive to unlearning target
    - Zone C (Conflict): Parameters shared by both retention and unlearning
    - Zone D (Plastic): Unused capacity for new learning
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        tau_r: float = 0.5,
        tau_f: float = 0.5,
        momentum_decay: float = 0.95,
        adaptation_rate: float = 0.01,
        surprise_threshold: float = 0.05,
        stabilization_decay: float = 0.3,
        device: str = 'cuda'
    ):
        """
        Initialize the LwU framework.
        
        Args:
            backbone: Neural network backbone (e.g., ResNet-18)
            num_classes: Total number of output classes
            tau_r: Threshold for retain importance (default: 0.5)
            tau_f: Threshold for forget importance (default: 0.5)
            momentum_decay: Momentum decay factor for TTU (varphi)
            adaptation_rate: Learning rate for TTU (lambda)
            surprise_threshold: Threshold for surprise gate (tau_s)
            stabilization_decay: Decay factor for parameter drift prevention (alpha)
            device: Device to use for computation
        """
        super().__init__()
        
        self.backbone = backbone
        self.num_classes = num_classes
        self.tau_r = tau_r
        self.tau_f = tau_f
        self.momentum_decay = momentum_decay
        self.adaptation_rate = adaptation_rate
        self.surprise_threshold = surprise_threshold
        self.stabilization_decay = stabilization_decay
        self.device = device
        
        # Initialize zone masks (binary masks for each parameter)
        self.zone_masks = {
            'A': {},  # Safe Retain
            'B': {},  # Pure Forget
            'C': {},  # Conflict
            'D': {}   # Plastic
        }
        
        # Accumulated mask from previous tasks
        self.accumulated_mask = {}
        
        # Importance scores
        self.phi_r = {}  # Retain importance (Shapley values)
        self.phi_f = {}  # Forget importance (Shapley values)
        
        # Momentum buffer for Test-Time Update
        self.momentum_buffer = {}
        
        # Original parameters for TTU restoration
        self.original_params = {}
        
        # Task counter
        self.current_task = 0
        
        # Initialize masks to zeros
        self._initialize_masks()
        
    def _initialize_masks(self):
        """Initialize all masks to zeros (all parameters in Zone D initially)."""
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                self.zone_masks['A'][name] = torch.zeros_like(param, dtype=torch.bool)
                self.zone_masks['B'][name] = torch.zeros_like(param, dtype=torch.bool)
                self.zone_masks['C'][name] = torch.zeros_like(param, dtype=torch.bool)
                self.zone_masks['D'][name] = torch.ones_like(param, dtype=torch.bool)
                self.accumulated_mask[name] = torch.zeros_like(param, dtype=torch.bool)
                self.momentum_buffer[name] = torch.zeros_like(param)
                
    def compute_fisher_information(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information for each parameter.
        
        Fisher Information estimates local sensitivity as the expected
        squared gradient magnitude (Equation 1 in the main text).
        
        Args:
            dataloader: DataLoader for the dataset
            criterion: Loss function (default: CrossEntropyLoss)
            
        Returns:
            Dictionary mapping parameter names to Fisher Information values
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
            
        fisher = {name: torch.zeros_like(param) 
                  for name, param in self.backbone.named_parameters() 
                  if param.requires_grad}
        
        self.backbone.eval()
        num_samples = 0
        
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.backbone.zero_grad()
            outputs = self.backbone(inputs)
            
            # Use log-likelihood for Fisher computation
            log_probs = F.log_softmax(outputs, dim=1)
            loss = F.nll_loss(log_probs, targets)
            loss.backward()
            
            for name, param in self.backbone.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2
                    
            num_samples += inputs.size(0)
            
        # Normalize by number of samples
        for name in fisher:
            fisher[name] /= num_samples
            
        self.backbone.train()
        return fisher
    
    def compute_ssv(
        self,
        dataloader: torch.utils.data.DataLoader,
        fisher: Dict[str, torch.Tensor],
        criterion: nn.Module = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Shapley Synaptic Values (SSV) using closed-form approximation.
        
        Under the diagonal Fisher approximation (F_ij = 0 for i != j), the
        closed-form SSV (Eq. 5 in the main text) simplifies to:
            phi_i = -g_i * theta_i + (1/2) * theta_i^2 * F_ii
        
        The first term captures individual importance (gradient-weight product),
        the second captures self-curvature via the diagonal Fisher entry.
        
        Args:
            dataloader: DataLoader for computing gradients
            fisher: Diagonal Fisher Information values for each parameter
            criterion: Loss function (default: CrossEntropyLoss)
            
        Returns:
            Dictionary mapping parameter names to SSV scores
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        # Compute average gradient over the dataset
        gradients = {name: torch.zeros_like(param) 
                     for name, param in self.backbone.named_parameters() 
                     if param.requires_grad}
        
        self.backbone.eval()
        num_samples = 0
        
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.backbone.zero_grad()
            outputs = self.backbone(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            for name, param in self.backbone.named_parameters():
                if param.requires_grad and param.grad is not None:
                    gradients[name] += param.grad.data * inputs.size(0)
                    
            num_samples += inputs.size(0)
        
        for name in gradients:
            gradients[name] /= num_samples
        
        self.backbone.train()
        
        # Compute SSV: phi_i = -g_i * theta_i + (1/2) * theta_i^2 * F_ii
        ssv = {}
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                first_order = -gradients[name] * param.data
                second_order = 0.5 * fisher[name] * (param.data ** 2)
                ssv[name] = first_order + second_order
                
        return ssv
    
    def identify_zones(
        self,
        retain_loader: torch.utils.data.DataLoader,
        forget_loader: Optional[torch.utils.data.DataLoader] = None
    ):
        """
        Identify the four topological zones based on importance scores.
        
        Zone A (Safe Retain): M_A = M_r ⊙ (1 - M_f)
        Zone B (Pure Forget): M_B = M_f ⊙ (1 - M_r)
        Zone C (Conflict): M_C = M_r ⊙ M_f
        Zone D (Plastic): M_D = (1 - M_r) ⊙ (1 - M_f)
        
        Args:
            retain_loader: DataLoader for retain set
            forget_loader: DataLoader for forget set (optional for CL-only mode)
        """
        # Compute Fisher Information and SSV for retain set
        fisher_r = self.compute_fisher_information(retain_loader)
        self.phi_r = self.compute_ssv(retain_loader, fisher_r)
        
        # Compute for forget set if provided
        if forget_loader is not None:
            fisher_f = self.compute_fisher_information(forget_loader)
            self.phi_f = self.compute_ssv(forget_loader, fisher_f)
        else:
            # For continual learning without unlearning, use zeros
            self.phi_f = {name: torch.zeros_like(param) 
                         for name, param in self.backbone.named_parameters() 
                         if param.requires_grad}
        
        # Create binary masks based on thresholds (Eq. 8: M_r = I[|phi_r| > tau_r])
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                # Threshold on absolute SSV scores, normalized to [0, 1]
                phi_r_abs = self._normalize_tensor(self.phi_r[name].abs())
                phi_f_abs = self._normalize_tensor(self.phi_f[name].abs())
                
                # Binary masks based on thresholds
                M_r = phi_r_abs > self.tau_r
                M_f = phi_f_abs > self.tau_f
                
                # Compute zone masks (Equations 5-8)
                self.zone_masks['A'][name] = M_r & ~M_f  # Safe Retain
                self.zone_masks['B'][name] = M_f & ~M_r  # Pure Forget
                self.zone_masks['C'][name] = M_r & M_f   # Conflict
                self.zone_masks['D'][name] = ~M_r & ~M_f  # Plastic
                
    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor to [0, 1] range."""
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val - min_val > 1e-8:
            return (tensor - min_val) / (max_val - min_val)
        return torch.zeros_like(tensor)
    
    def apply_zone_operations(
        self,
        retain_loader: Optional[torch.utils.data.DataLoader] = None,
        forget_loader: Optional[torch.utils.data.DataLoader] = None,
        eta_c: float = 0.01,
        n_c: int = 5,
        delta_c: float = 0.1
    ):
        """
        Apply zone-specific operations after zone identification.
        
        - Zone A: Frozen (handled during training via gradient masking)
        - Zone B: Reinitialize with He initialization (Eq. 12)
        - Zone C: Orthogonal conflict resolution via projected gradient ascent (Eqs. 13-17)
        - Zone D: Available for learning (no modification needed)
        
        Args:
            retain_loader: DataLoader for retain set (needed for Zone C)
            forget_loader: DataLoader for forget set (needed for Zone C)
            eta_c: Learning rate for Zone C conflict resolution
            n_c: Number of projected gradient ascent steps
            delta_c: Displacement budget for Zone C
        """
        criterion = nn.CrossEntropyLoss()
        
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                # Zone B: Reinitialize (Eq. 12)
                mask_B = self.zone_masks['B'][name]
                if mask_B.any():
                    fan_in = param.numel() / param.shape[0] if len(param.shape) > 1 else param.numel()
                    std = np.sqrt(2.0 / fan_in)
                    new_values = torch.randn_like(param) * std
                    param.data = torch.where(mask_B, new_values, param.data)
        
        # Zone C: Orthogonal Conflict Resolution (Eqs. 13-17)
        if retain_loader is not None and forget_loader is not None:
            self._orthogonal_conflict_resolution(
                retain_loader, forget_loader, criterion, eta_c, n_c, delta_c
            )
                    
        # Update accumulated mask
        self._update_accumulated_mask()
    
    def _orthogonal_conflict_resolution(
        self,
        retain_loader: torch.utils.data.DataLoader,
        forget_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        eta_c: float,
        n_c: int,
        delta_c: float
    ):
        """
        Zone C: Orthogonal conflict resolution via projected gradient ascent.
        
        Computes the forget gradient g_f^C and retain gradient g_r^C on Zone C
        parameters, then performs gradient ascent on the forget loss using the
        component of g_f^C orthogonal to g_r^C (Eqs. 13-17).
        
        Args:
            retain_loader: DataLoader for retain set
            forget_loader: DataLoader for forget set
            criterion: Loss function
            eta_c: Learning rate for conflict resolution
            n_c: Number of projected gradient ascent steps
            delta_c: Displacement budget
        """
        # Store initial Zone C parameters for displacement clipping
        theta_c_init = {}
        for name, param in self.backbone.named_parameters():
            if param.requires_grad and self.zone_masks['C'][name].any():
                theta_c_init[name] = param.data.clone()
        
        if not theta_c_init:
            return
        
        for step in range(n_c):
            # Compute retain gradient on Zone C
            g_r = self._compute_zone_c_gradient(retain_loader, criterion)
            
            # Compute forget gradient on Zone C
            g_f = self._compute_zone_c_gradient(forget_loader, criterion)
            
            # For each parameter, project and update
            for name, param in self.backbone.named_parameters():
                if name not in theta_c_init:
                    continue
                    
                mask_C = self.zone_masks['C'][name].float()
                g_r_c = g_r[name] * mask_C
                g_f_c = g_f[name] * mask_C
                
                # Eq. 13: Unlearning direction = g_f^C (gradient ascent on forget loss)
                g_u = g_f_c
                
                # Eq. 14: Orthogonal projection
                dot = (g_u * g_r_c).sum()
                norm_sq = (g_r_c * g_r_c).sum() + 1e-8
                g_u_perp = g_u - (dot / norm_sq) * g_r_c
                
                # Eq. 15: Projected gradient ascent step
                param.data = param.data + eta_c * g_u_perp
                
                # Eq. 17: Displacement clipping
                displacement = param.data - theta_c_init[name]
                disp_norm = displacement.norm()
                if disp_norm > delta_c:
                    param.data = theta_c_init[name] + displacement * (delta_c / disp_norm)
    
    def _compute_zone_c_gradient(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """Compute average gradient restricted to Zone C parameters."""
        grads = {name: torch.zeros_like(param) 
                 for name, param in self.backbone.named_parameters() 
                 if param.requires_grad}
        
        self.backbone.eval()
        num_samples = 0
        
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.backbone.zero_grad()
            outputs = self.backbone(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            for name, param in self.backbone.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grads[name] += param.grad.data * inputs.size(0)
            
            num_samples += inputs.size(0)
        
        for name in grads:
            grads[name] /= num_samples
        
        self.backbone.train()
        return grads
        
    def _update_accumulated_mask(self):
        """Update accumulated mask after zone operations (Eq. 20)."""
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                # Release Zone B and conflict-resolved Zone C parameters
                released = self.zone_masks['B'][name].clone()
                released = released | self.zone_masks['C'][name]
                
                # Remove released parameters from accumulated mask
                self.accumulated_mask[name] = self.accumulated_mask[name] & ~released
                
                # Add current retain mask (Zone A)
                self.accumulated_mask[name] = self.accumulated_mask[name] | self.zone_masks['A'][name]
                
    def get_trainable_mask(self) -> Dict[str, torch.Tensor]:
        """
        Get mask of parameters that can be updated during training.
        
        Returns mask where 1 = can update, 0 = frozen
        """
        trainable_mask = {}
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                # Can train: Zone D + Zone B + released Zone C
                trainable_mask[name] = ~self.accumulated_mask[name]
        return trainable_mask
    
    def masked_backward(self, loss: torch.Tensor):
        """
        Perform backward pass with gradient masking.
        
        Gradients are zeroed for parameters in Zone A (accumulated mask).
        
        Args:
            loss: Loss tensor to backpropagate
        """
        loss.backward()
        
        for name, param in self.backbone.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Zero gradients for protected parameters
                param.grad.data = param.grad.data * (~self.accumulated_mask[name]).float()
                
    def test_time_update(
        self,
        inputs: torch.Tensor,
        use_adaptation: bool = True
    ) -> torch.Tensor:
        """
        Perform Test-Time Update (TTU) using entropy minimization.
        
        Adapts Zone A parameters during inference using a momentum-based
        update with a surprise gate (Equations 9-11 in the main text).
        
        Args:
            inputs: Input tensor
            use_adaptation: Whether to apply TTU
            
        Returns:
            Model predictions
        """
        if not use_adaptation:
            return self.backbone(inputs)
            
        # Store original parameters if not already stored
        if not self.original_params:
            for name, param in self.backbone.named_parameters():
                if param.requires_grad:
                    self.original_params[name] = param.data.clone()
        
        self.backbone.eval()
        
        # Enable gradient computation for TTU
        with torch.enable_grad():
            inputs = inputs.clone().detach().requires_grad_(False)
            outputs = self.backbone(inputs)
            
            # Compute entropy loss (Equation 9)
            probs = F.softmax(outputs, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            
            # Compute gradients
            self.backbone.zero_grad()
            entropy.backward()
            
            # Apply TTU to Zone A parameters
            for name, param in self.backbone.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Equation 10: g^t_A = ∇_θ ℓ_ent(x_t) ⊙ M_A
                    mask_A = self.zone_masks['A'][name].float()
                    g_A = param.grad.data * mask_A
                    
                    # Equation 11: S^t_A = φ * S^{t-1}_A - λ * I[||g^t_A|| > τ_s] * g^t_A
                    # The indicator function gates the gradient contribution, NOT the entire update
                    # Momentum decay ALWAYS happens; gradient contribution is gated
                    gate = 1.0 if g_A.norm() > self.surprise_threshold else 0.0
                    self.momentum_buffer[name] = (
                        self.momentum_decay * self.momentum_buffer[name] 
                        - self.adaptation_rate * gate * g_A
                    )
                    
                    # Eq. 16: θ_{A,test}^{t+1} = (1-α) * θ_{A,test}^t + α * θ_A + S^t_A
                    # EMA anchors back to frozen pretrained weights, preventing drift to zero
                    param.data = (
                        (1 - self.stabilization_decay) * param.data 
                        + self.stabilization_decay * self.original_params[name]
                        + self.momentum_buffer[name]
                    )
        
        # Get final prediction
        with torch.no_grad():
            outputs = self.backbone(inputs)
            
        return outputs
    
    def restore_original_params(self):
        """Restore parameters to their pre-TTU state."""
        if self.original_params:
            for name, param in self.backbone.named_parameters():
                if name in self.original_params:
                    param.data = self.original_params[name].clone()
        self.original_params = {}
        
        # Reset momentum buffer
        for name in self.momentum_buffer:
            self.momentum_buffer[name].zero_()
            
    def forward(
        self,
        x: torch.Tensor,
        test_time_adapt: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            test_time_adapt: Whether to use Test-Time Update
            
        Returns:
            Model predictions
        """
        if test_time_adapt and self.training is False:
            return self.test_time_update(x)
        return self.backbone(x)
    
    def new_task(
        self,
        retain_loader: torch.utils.data.DataLoader,
        forget_loader: Optional[torch.utils.data.DataLoader] = None,
        eta_c: float = 0.01,
        n_c: int = 5,
        delta_c: float = 0.1
    ):
        """
        Prepare for learning a new task.
        
        This method should be called before training on each new task.
        It identifies zones and applies zone-specific operations.
        
        Args:
            retain_loader: DataLoader for data to retain
            forget_loader: DataLoader for data to forget (optional)
            eta_c: Learning rate for Zone C conflict resolution
            n_c: Number of Zone C projected gradient ascent steps
            delta_c: Displacement budget for Zone C
        """
        self.current_task += 1
        
        # Identify zones based on importance
        self.identify_zones(retain_loader, forget_loader)
        
        # Apply zone-specific operations
        self.apply_zone_operations(retain_loader, forget_loader, eta_c, n_c, delta_c)
        
    def get_zone_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics about zone distributions.
        
        Returns:
            Dictionary with zone statistics per layer and overall
        """
        stats = {'overall': {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'total': 0}}
        
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                total = param.numel()
                count_A = self.zone_masks['A'][name].sum().item()
                count_B = self.zone_masks['B'][name].sum().item()
                count_C = self.zone_masks['C'][name].sum().item()
                count_D = self.zone_masks['D'][name].sum().item()
                
                stats[name] = {
                    'A': count_A / total * 100,
                    'B': count_B / total * 100,
                    'C': count_C / total * 100,
                    'D': count_D / total * 100,
                    'total': total
                }
                
                stats['overall']['A'] += count_A
                stats['overall']['B'] += count_B
                stats['overall']['C'] += count_C
                stats['overall']['D'] += count_D
                stats['overall']['total'] += total
                
        # Convert overall to percentages
        total = stats['overall']['total']
        if total > 0:
            stats['overall']['A'] = stats['overall']['A'] / total * 100
            stats['overall']['B'] = stats['overall']['B'] / total * 100
            stats['overall']['C'] = stats['overall']['C'] / total * 100
            stats['overall']['D'] = stats['overall']['D'] / total * 100
            
        return stats
    
    def unlearn(
        self,
        forget_loader: torch.utils.data.DataLoader,
        retain_loader: torch.utils.data.DataLoader,
        eta_c: float = 0.01,
        n_c: int = 5,
        delta_c: float = 0.1
    ):
        """
        Perform machine unlearning on the specified forget set.
        
        Args:
            forget_loader: DataLoader for data to forget
            retain_loader: DataLoader for data to retain
            eta_c: Learning rate for Zone C conflict resolution
            n_c: Number of Zone C projected gradient ascent steps
            delta_c: Displacement budget for Zone C
        """
        # Identify zones with both retain and forget sets
        self.identify_zones(retain_loader, forget_loader)
        
        # Apply zone operations (reinitialize Zone B, resolve conflicts in Zone C)
        self.apply_zone_operations(retain_loader, forget_loader, eta_c, n_c, delta_c)
