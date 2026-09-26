"""
Learning with Unlearning (LwU) - four-zone parameter decomposition (base stage)

A unified framework for simultaneous Continual Learning and Machine Unlearning
through principled parameter space decomposition into four topological zones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from typing import Dict, Optional


class ZoneDecomposition(nn.Module):
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
        threshold_mode: str = 'absolute',
        device: str = 'cuda',
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
        if threshold_mode not in {'absolute', 'normalized'}:
            raise ValueError("threshold_mode must be 'absolute' or 'normalized'")
        self.threshold_mode = threshold_mode
        self.device = device

        # Initialize zone masks (binary masks for each parameter)
        self.zone_masks = {
            'A': {},  # Safe Retain
            'B': {},  # Pure Forget
            'C': {},  # Conflict
            'D': {},  # Plastic
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

        # Zone-C parameters are released only when conflict resolution really
        # updated them.  This is distinct from the full Zone-C mask.
        self.released_c_masks = {}

        # Task counter
        self.current_task = 0
        self.last_zone_c_budget = 0.0
        self.diagnostics = {}

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
                self.released_c_masks[name] = torch.zeros_like(param, dtype=torch.bool)

    def get_extra_state(self):
        """Persist the non-parameter continual/unlearning state in checkpoints."""
        return {
            'zone_masks': {
                z: {n: t.detach().cpu() for n, t in masks.items()}
                for z, masks in self.zone_masks.items()
            },
            'accumulated_mask': {n: t.detach().cpu() for n, t in self.accumulated_mask.items()},
            'released_c_masks': {n: t.detach().cpu() for n, t in self.released_c_masks.items()},
            'current_task': self.current_task,
            'threshold_mode': self.threshold_mode,
        }

    def set_extra_state(self, state):
        if not state:
            return
        parameters = dict(self.backbone.named_parameters())
        for zone, masks in state.get('zone_masks', {}).items():
            for name, tensor in masks.items():
                if name in parameters:
                    self.zone_masks[zone][name] = tensor.to(parameters[name].device)
        for field in ('accumulated_mask', 'released_c_masks'):
            destination = getattr(self, field)
            for name, tensor in state.get(field, {}).items():
                if name in parameters:
                    destination[name] = tensor.to(parameters[name].device)
        self.current_task = int(state.get('current_task', 0))
        self.threshold_mode = state.get('threshold_mode', self.threshold_mode)

    @staticmethod
    def _require_nonempty(num_samples: int, name: str):
        if num_samples == 0:
            raise ValueError(f'{name} loader is empty')

    def compute_fisher_information(
        self, dataloader: torch.utils.data.DataLoader, criterion: nn.Module = None
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

        fisher = {
            name: torch.zeros_like(param)
            for name, param in self.backbone.named_parameters()
            if param.requires_grad
        }

        was_training = self.backbone.training
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
                    # loss is a batch mean, so weight the batch-gradient
                    # estimate by its number of observations before taking the
                    # expectation over the loader.
                    fisher[name] += (param.grad.detach() ** 2) * inputs.size(0)

            num_samples += inputs.size(0)

        self._require_nonempty(num_samples, 'Fisher')
        for name in fisher:
            fisher[name] /= num_samples

        self.backbone.train(was_training)
        return fisher

    def compute_ssv(
        self,
        dataloader: torch.utils.data.DataLoader,
        fisher: Dict[str, torch.Tensor],
        criterion: nn.Module = None,
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
        gradients = {
            name: torch.zeros_like(param)
            for name, param in self.backbone.named_parameters()
            if param.requires_grad
        }

        was_training = self.backbone.training
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
                    gradients[name] += param.grad.detach() * inputs.size(0)

            num_samples += inputs.size(0)

        self._require_nonempty(num_samples, 'SSV')
        for name in gradients:
            gradients[name] /= num_samples

        self.backbone.train(was_training)

        # Compute SSV: phi_i = -g_i * theta_i + (1/2) * theta_i^2 * F_ii
        ssv = {}
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                first_order = -gradients[name] * param.data
                second_order = 0.5 * fisher[name] * (param.data**2)
                ssv[name] = first_order + second_order

        return ssv

    def compute_importance(
        self, dataloader: torch.utils.data.DataLoader, criterion: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """Estimate gradient, diagonal empirical Fisher, and SSV in one pass.

        The manuscript states that the diagonal approximation needs one
        retain and one forget forward/backward pass.  This routine avoids the
        old implementation's second traversal of each partition.  For a
        mini-batched loader, each batch contributes a sample-count-weighted
        gradient and squared batch-gradient estimate.
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        gradients = {
            n: torch.zeros_like(p) for n, p in self.backbone.named_parameters() if p.requires_grad
        }
        fisher = {
            n: torch.zeros_like(p) for n, p in self.backbone.named_parameters() if p.requires_grad
        }
        was_training = self.backbone.training
        self.backbone.eval()
        num_samples = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.backbone.zero_grad(set_to_none=True)
            criterion(self.backbone(inputs), targets).backward()
            batch_n = inputs.size(0)
            for name, param in self.backbone.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad = param.grad.detach()
                    gradients[name].add_(grad, alpha=batch_n)
                    fisher[name].add_(grad.square(), alpha=batch_n)
            num_samples += batch_n
        self._require_nonempty(num_samples, 'importance')
        result = {}
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                gradient = gradients[name] / num_samples
                fisher[name].div_(num_samples)
                result[name] = (
                    -gradient * param.detach() + 0.5 * param.detach().square() * fisher[name]
                )
        self.backbone.zero_grad(set_to_none=True)
        self.backbone.train(was_training)
        return result

    def compute_ssv_with_interaction(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: Optional[nn.Module] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the full empirical-Fisher SSV without storing dense Fisher.

        For the empirical Fisher ``F = E[g g^T]``, the interaction term only
        needs ``F theta``.  A minibatch gradient therefore contributes
        ``g * (g dot theta)``.  This evaluates

            -g_i theta_i + 1/2 theta_i^2 F_ii
            + 1/2 theta_i sum_{j != i} F_ij theta_j

        with three model-sized accumulators instead of an N-by-N matrix.
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        parameters = {
            name: parameter
            for name, parameter in self.backbone.named_parameters()
            if parameter.requires_grad
        }
        gradients = {name: torch.zeros_like(parameter) for name, parameter in parameters.items()}
        fisher_diagonal = {
            name: torch.zeros_like(parameter) for name, parameter in parameters.items()
        }
        fisher_theta = {name: torch.zeros_like(parameter) for name, parameter in parameters.items()}
        was_training = self.backbone.training
        self.backbone.eval()
        num_samples = 0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.backbone.zero_grad(set_to_none=True)
            criterion(self.backbone(inputs), targets).backward()
            batch_n = inputs.size(0)
            theta_dot_gradient = torch.zeros((), device=inputs.device)
            for name, parameter in parameters.items():
                if parameter.grad is not None:
                    theta_dot_gradient.add_((parameter.grad.detach() * parameter.detach()).sum())
            for name, parameter in parameters.items():
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.detach()
                gradients[name].add_(gradient, alpha=batch_n)
                fisher_diagonal[name].add_(gradient.square(), alpha=batch_n)
                fisher_theta[name].add_(gradient * theta_dot_gradient, alpha=batch_n)
            num_samples += batch_n

        self._require_nonempty(num_samples, 'interaction SSV')
        result = {}
        for name, parameter in parameters.items():
            theta = parameter.detach()
            mean_gradient = gradients[name] / num_samples
            fisher_diagonal[name].div_(num_samples)
            fisher_theta[name].div_(num_samples)
            interaction = fisher_theta[name] - fisher_diagonal[name] * theta
            result[name] = (
                -mean_gradient * theta
                + 0.5 * theta.square() * fisher_diagonal[name]
                + 0.5 * theta * interaction
            )
        self.backbone.zero_grad(set_to_none=True)
        self.backbone.train(was_training)
        return result

    def identify_zones(
        self,
        retain_loader: torch.utils.data.DataLoader,
        forget_loader: Optional[torch.utils.data.DataLoader] = None,
        criterion: Optional[nn.Module] = None,
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
        self.phi_r = self.compute_importance(retain_loader, criterion)

        # Compute for forget set if provided
        if forget_loader is not None:
            self.phi_f = self.compute_importance(forget_loader, criterion)
        else:
            # For continual learning without unlearning, use zeros
            self.phi_f = {
                name: torch.zeros_like(param)
                for name, param in self.backbone.named_parameters()
                if param.requires_grad
            }

        # Create binary masks based on thresholds (Eq. 8: M_r = I[|phi_r| > tau_r])
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                # Eq. (masks) thresholds the absolute SSV itself.  The previous
                # code silently min-max normalized each tensor, which changes
                # the equation and makes tau incomparable across layers.
                phi_r_abs = self.phi_r[name].abs()
                phi_f_abs = self.phi_f[name].abs()
                if self.threshold_mode == 'normalized':
                    # Kept only as an explicit ablation for legacy configs.
                    phi_r_abs = self._normalize_tensor(phi_r_abs)
                    phi_f_abs = self._normalize_tensor(phi_f_abs)

                # Binary masks based on thresholds
                M_r = phi_r_abs > self.tau_r
                M_f = phi_f_abs > self.tau_f

                # Compute zone masks (Equations 5-8)
                self.zone_masks['A'][name] = M_r & ~M_f  # Safe Retain
                self.zone_masks['B'][name] = M_f & ~M_r  # Pure Forget
                self.zone_masks['C'][name] = M_r & M_f  # Conflict
                self.zone_masks['D'][name] = ~M_r & ~M_f  # Plastic

    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize tensor to [0, 1] range."""
        min_val = tensor.min()
        max_val = tensor.max()
        if max_val - min_val > 1e-8:
            return (tensor - min_val) / (max_val - min_val)
        return torch.zeros_like(tensor)

    @torch.no_grad()
    def zone_reconstruction_diagnostics(self) -> Dict[str, float]:
        """Check that A/B/C/D are disjoint, exhaustive, and lossless."""
        max_error = 0.0
        overlap_or_gap = 0
        total = 0
        for name, parameter in self.backbone.named_parameters():
            if not parameter.requires_grad:
                continue
            masks = [self.zone_masks[zone][name] for zone in 'ABCD']
            membership = torch.stack(masks).sum(dim=0)
            overlap_or_gap += int(membership.ne(1).sum().item())
            reconstructed = sum(parameter * mask for mask in masks)
            max_error = max(max_error, float((reconstructed - parameter).abs().max().item()))
            total += parameter.numel()
        return {
            'max_abs_reconstruction_error': max_error,
            'overlap_or_gap_parameters': overlap_or_gap,
            'trainable_parameters': total,
        }

    def apply_zone_operations(
        self,
        retain_loader: Optional[torch.utils.data.DataLoader] = None,
        forget_loader: Optional[torch.utils.data.DataLoader] = None,
        eta_c: float = 0.01,
        n_c: int = 5,
        delta_c: float = 0.1,
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
                    new_values = torch.empty_like(param)
                    if param.ndim >= 2:
                        nn.init.kaiming_normal_(new_values, mode='fan_in', nonlinearity='relu')
                    else:
                        nn.init.zeros_(new_values)
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
        delta_c: float,
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
        # Store initial Zone C parameters for displacement clipping.
        theta_c_init = {}
        for name, param in self.backbone.named_parameters():
            if param.requires_grad and self.zone_masks['C'][name].any():
                theta_c_init[name] = param.data.clone()

        if not theta_c_init:
            return

        # The paper defines delta_C as proportional to average Zone-C SSV.
        zone_scores = []
        for name in theta_c_init:
            mask = self.zone_masks['C'][name]
            zone_scores.append(0.5 * (self.phi_r[name].abs()[mask] + self.phi_f[name].abs()[mask]))
        mean_zone_ssv = torch.cat(zone_scores).mean()
        budget = float(delta_c * mean_zone_ssv.item())
        self.last_zone_c_budget = budget

        for name in self.released_c_masks:
            self.released_c_masks[name].zero_()

        for _ in range(n_c):
            # Compute retain gradient on Zone C
            g_r = self._compute_zone_c_gradient(retain_loader, criterion)

            # Compute forget gradient on Zone C
            g_f = self._compute_zone_c_gradient(forget_loader, criterion)

            # Eq. (projection) is defined over the complete theta_C vector,
            # not independently per layer.  Use one global projection scalar.
            dot = torch.zeros((), device=next(self.backbone.parameters()).device)
            norm_sq = torch.zeros_like(dot)
            masked = {}
            for name in theta_c_init:
                mask = self.zone_masks['C'][name]
                gr = g_r[name] * mask
                gf = g_f[name] * mask
                masked[name] = (gr, gf)
                dot += (gf * gr).sum()
                norm_sq += gr.square().sum()
            coefficient = dot / (norm_sq + 1e-8)
            for name, param in self.backbone.named_parameters():
                if name in theta_c_init:
                    gr, gf = masked[name]
                    update = (gf - coefficient * gr) * self.zone_masks['C'][name]
                    param.data.add_(update, alpha=eta_c)

            # Clip the cumulative displacement as one Zone-C vector.
            disp_sq = torch.zeros_like(dot)
            for name, param in self.backbone.named_parameters():
                if name in theta_c_init:
                    disp = (param.data - theta_c_init[name]) * self.zone_masks['C'][name]
                    disp_sq += disp.square().sum()
            disp_norm = disp_sq.sqrt()
            if disp_norm > budget:
                scale = budget / (disp_norm + 1e-12)
                for name, param in self.backbone.named_parameters():
                    if name in theta_c_init:
                        mask = self.zone_masks['C'][name]
                        clipped = theta_c_init[name] + (param.data - theta_c_init[name]) * scale
                        param.data.copy_(torch.where(mask, clipped, param.data))

        # Release only conflict weights with a non-zero *final* displacement.
        # Marking the pre-clipping direction released weights even when a zero
        # budget restored them exactly to theta_C^(0).
        for name, param in self.backbone.named_parameters():
            if name in theta_c_init:
                mask = self.zone_masks['C'][name]
                displacement = (param.data - theta_c_init[name]).abs()
                self.released_c_masks[name] = mask & displacement.gt(1e-12)

    def _compute_zone_c_gradient(
        self, dataloader: torch.utils.data.DataLoader, criterion: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """Compute average gradient restricted to Zone C parameters."""
        grads = {
            name: torch.zeros_like(param)
            for name, param in self.backbone.named_parameters()
            if param.requires_grad
        }

        was_training = self.backbone.training
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

        self._require_nonempty(num_samples, 'Zone-C gradient')
        for name in grads:
            grads[name] /= num_samples

        self.backbone.train(was_training)
        return grads

    def _update_accumulated_mask(self):
        """Update accumulated mask after zone operations (Eq. 20)."""
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                # Release Zone B and conflict-resolved Zone C parameters
                released = self.zone_masks['B'][name] | self.released_c_masks[name]
                retain = self.zone_masks['A'][name] | self.zone_masks['C'][name]
                self.accumulated_mask[name] = (self.accumulated_mask[name] | retain) & ~released

    def consolidate_task(
        self,
        task_loader: torch.utils.data.DataLoader,
        forget_loader: Optional[torch.utils.data.DataLoader] = None,
        criterion: Optional[nn.Module] = None,
    ):
        """Value and freeze the just-learned task.

        With ``forget_loader=None`` (the default) this is the buffer-free
        continual-learning transition: the caller invokes it while the
        current task is still available, after training that task and before
        discarding its loader. Passing a ``forget_loader`` is an explicit
        ablation opt-in that feeds a real forget-side importance signal into
        estimators that would otherwise see all-zero forget scores.
        """
        self.identify_zones(task_loader, forget_loader=forget_loader, criterion=criterion)
        self._update_accumulated_mask()
        self.current_task += 1

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

    def test_time_update(self, inputs: torch.Tensor, use_adaptation: bool = True) -> torch.Tensor:
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

            # Eq. (surprise gate) takes the norm of the complete Zone-A
            # gradient vector.  A per-layer gate changes the method because it
            # can adapt some layers and suppress others for the same sample.
            masked_gradients = {}
            zone_a_norm_sq = torch.zeros((), device=inputs.device)
            for name, param in self.backbone.named_parameters():
                if param.requires_grad and param.grad is not None:
                    g_A = param.grad.detach() * self.zone_masks['A'][name]
                    masked_gradients[name] = g_A
                    zone_a_norm_sq += g_A.square().sum()
            gate = float(zone_a_norm_sq.sqrt() > self.surprise_threshold)

            # Apply TTU to Zone A parameters
            for name, param in self.backbone.named_parameters():
                if name in masked_gradients:
                    g_A = masked_gradients[name]
                    # Equation 11: S^t_A = φ * S^{t-1}_A - λ * I[||g^t_A|| > τ_s] * g^t_A
                    # The indicator function gates the gradient contribution, NOT the entire update
                    # Momentum decay ALWAYS happens; gradient contribution is gated
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

    @contextmanager
    def test_stream(self):
        """Guarantee that inference-time TTU state is ephemeral."""
        try:
            yield self
        finally:
            self.restore_original_params()

    def forward(self, x: torch.Tensor, test_time_adapt: bool = False) -> torch.Tensor:
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

    def effective_parameter_tensors(self):
        """Expose masked effective weights for distributional comparisons."""
        if hasattr(self.backbone, 'effective_parameter_tensors'):
            yield from self.backbone.effective_parameter_tensors()
        else:
            yield from self.backbone.parameters()

    def new_task(
        self,
        retain_loader: torch.utils.data.DataLoader,
        forget_loader: Optional[torch.utils.data.DataLoader] = None,
        eta_c: float = 0.01,
        n_c: int = 5,
        delta_c: float = 0.1,
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
        stats = {
            'overall': {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'total': 0},
            'accumulated': {'protected': 0, 'plastic': 0, 'total': 0},
        }

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
                    'total': total,
                }

                stats['overall']['A'] += count_A
                stats['overall']['B'] += count_B
                stats['overall']['C'] += count_C
                stats['overall']['D'] += count_D
                stats['overall']['total'] += total
                protected = self.accumulated_mask[name].sum().item()
                stats['accumulated']['protected'] += protected
                stats['accumulated']['plastic'] += total - protected
                stats['accumulated']['total'] += total

        # Convert overall to percentages
        total = stats['overall']['total']
        if total > 0:
            stats['overall']['A'] = stats['overall']['A'] / total * 100
            stats['overall']['B'] = stats['overall']['B'] / total * 100
            stats['overall']['C'] = stats['overall']['C'] / total * 100
            stats['overall']['D'] = stats['overall']['D'] / total * 100

        accumulated_total = stats['accumulated']['total']
        if accumulated_total > 0:
            stats['accumulated']['protected'] = (
                stats['accumulated']['protected'] / accumulated_total * 100
            )
            stats['accumulated']['plastic'] = (
                stats['accumulated']['plastic'] / accumulated_total * 100
            )

        return stats

    def unlearn(
        self,
        forget_loader: torch.utils.data.DataLoader,
        retain_loader: torch.utils.data.DataLoader,
        eta_c: float = 0.01,
        n_c: int = 5,
        delta_c: float = 0.1,
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
        reconstruction = self.zone_reconstruction_diagnostics()

        # Apply zone operations (reinitialize Zone B, resolve conflicts in Zone C)
        self.apply_zone_operations(retain_loader, forget_loader, eta_c, n_c, delta_c)
        self.diagnostics = {
            'importance': 'ssv',
            'threshold_mode': self.threshold_mode,
            'tau_r': self.tau_r,
            'tau_f': self.tau_f,
            'reconstruction': reconstruction,
            'zones_pct': self.get_zone_statistics()['overall'],
            'zone_c_budget': self.last_zone_c_budget,
        }
