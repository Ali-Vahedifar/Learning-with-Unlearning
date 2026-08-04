"""
Evaluation Metrics for Continual Learning and Machine Unlearning

Implements metrics including:
- Average Accuracy (ACC)
- Backward Transfer (BWT)
- Forward Transfer (FWT)
- Plasticity-Stability (PS)
- Membership Inference Attack (MIA)
- KL Divergence for unlearning verification
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, roc_auc_score
import time


class AccuracyMatrix:
    """
    Maintains the accuracy matrix A for continual learning evaluation.
    
    A[i,j] = accuracy on task j after training on task i
    """
    
    def __init__(self, num_tasks: int):
        """
        Initialize accuracy matrix.
        
        Args:
            num_tasks: Total number of tasks
        """
        self.num_tasks = num_tasks
        self.matrix = np.zeros((num_tasks, num_tasks))
        self.random_acc = np.zeros(num_tasks)  # Random model accuracy per task
        self._pre_task_recorded = set()  # tasks with A[t-1, t] measured
        
    def update(
        self,
        current_task: int,
        task_accuracies: Dict[int, float]
    ):
        """
        Update accuracy matrix after training on current task.
        
        Args:
            current_task: Index of task just completed (0-indexed)
            task_accuracies: Dictionary mapping task_id to accuracy
        """
        for task_id, acc in task_accuracies.items():
            if task_id <= current_task:
                self.matrix[current_task, task_id] = acc
                
    def set_random_accuracy(self, task_id: int, random_acc: float):
        """Set random model accuracy for a task."""
        self.random_acc[task_id] = random_acc
        
    def get_acc(self) -> float:
        """
        Compute Average Accuracy (ACC).
        
        ACC = (1/K) * sum_{j=1}^{K} A[K,j]
        
        Returns:
            Average accuracy after learning all tasks
        """
        K = self.num_tasks
        return np.mean(self.matrix[K-1, :K])
    
    def get_bwt(self) -> float:
        """
        Compute Backward Transfer (BWT).
        
        BWT = (1/(K-1)) * sum_{j=1}^{K-1} (A[K,j] - A[j,j])
        
        Measures how learning new tasks affects performance on previous tasks.
        Negative BWT indicates forgetting.
        
        Returns:
            Backward transfer metric
        """
        K = self.num_tasks
        if K <= 1:
            return 0.0
            
        bwt = 0.0
        for j in range(K - 1):
            bwt += self.matrix[K-1, j] - self.matrix[j, j]
            
        return bwt / (K - 1)
    
    def get_fwt(self) -> float:
        """
        Compute Forward Transfer (FWT).
        
        FWT = (1/(K-1)) * sum_{j=2}^{K} (A[j-1,j] - RAC[j])
        
        Measures zero-shot transfer to future tasks.
        RAC = Random model accuracy
        
        Returns:
            Forward transfer metric
        """
        K = self.num_tasks
        if K <= 1:
            return 0.0
            
        fwt = 0.0
        for j in range(1, K):
            # A[j-1, j] is accuracy on task j before training on task j
            # This requires evaluating on unseen tasks, which we approximate
            # as the accuracy right after training on task j-1
            fwt += self.matrix[j-1, j] - self.random_acc[j]
            
        return fwt / (K - 1)
    
    def record_pre_task_accuracy(self, task_id: int, accuracy: float):
        """
        Record accuracy on task `task_id` measured BEFORE training on it, i.e.
        the matrix entry A[task_id - 1, task_id].

        This upper-diagonal entry is required by both FWT and the plasticity
        term of PS. It is not produced by `update`, which only fills entries
        for tasks already seen, so it must be recorded explicitly by calling
        `ContinualLearningEvaluator.evaluate_before_task` at the start of each
        task.
        """
        if task_id <= 0 or task_id >= self.num_tasks:
            return
        self.matrix[task_id - 1, task_id] = accuracy
        self._pre_task_recorded.add(task_id)

    def get_plasticity(self) -> float:
        """
        Plasticity: normalised learning gain on each new task.

            P = 1/(T-1) * sum_{t=2}^{T} (A[t,t] - A[t-1,t]) / (1 - A[t-1,t])

        The denominator normalises the realised gain by the gain that was
        available, so a task that was already partly solved before training
        is not credited with the same plasticity as one learned from scratch.
        """
        K = self.num_tasks
        if K <= 1:
            return float(self.matrix[0, 0]) if K == 1 else 0.0

        missing = [t for t in range(1, K) if t not in self._pre_task_recorded]
        if missing:
            raise ValueError(
                "Plasticity requires pre-task accuracies A[t-1, t] for tasks "
                f"{missing}, which were never recorded. Call "
                "ContinualLearningEvaluator.evaluate_before_task(t) at the "
                "start of each task, or compute PS only from runs that did. "
                "These entries cannot be recovered after the fact from a "
                "lower-triangular accuracy matrix."
            )

        gains = []
        for t in range(1, K):
            pre = float(self.matrix[t - 1, t])
            post = float(self.matrix[t, t])
            headroom = 1.0 - pre
            if headroom <= 1e-8:
                # No headroom left to measure; the task was already solved.
                gains.append(0.0)
            else:
                gains.append((post - pre) / headroom)

        return float(np.mean(gains))

    def get_stability(self) -> float:
        """
        Stability: one minus mean forgetting.

            S = 1 - 1/(T-1) * sum_{t=1}^{T-1} (A[t,t] - A[T,t])

        By construction S = 1 + BWT, since BWT is the mean of (A[T,t] - A[t,t])
        over the same range.
        """
        K = self.num_tasks
        if K <= 1:
            return 1.0

        forgetting = [
            float(self.matrix[t, t]) - float(self.matrix[K - 1, t])
            for t in range(K - 1)
        ]
        return 1.0 - float(np.mean(forgetting))

    def get_ps(self) -> float:
        """
        Compute the Plasticity-Stability (PS) metric as the harmonic mean

            PS = 2 * P * S / (P + S)

        with P and S as defined in `get_plasticity` and `get_stability`.

        NOTE ON PROVENANCE: an earlier version of this file computed P as the
        mean of the accuracy-matrix diagonal and S as a mean retention ratio,
        following the reference implementation of the work that introduced PS.
        That is not the definition stated in this paper. The definition above
        is the paper's, and it is the one this file now computes.
        """
        K = self.num_tasks
        if K <= 1:
            return float(self.matrix[0, 0]) if K == 1 else 0.0

        plasticity = self.get_plasticity()
        stability = self.get_stability()

        if plasticity + stability <= 0:
            return 0.0

        return 2.0 * plasticity * stability / (plasticity + stability)
    
    def get_all_metrics(self) -> Dict[str, float]:
        """Get all continual learning metrics."""
        return {
            'ACC': self.get_acc(),
            'BWT': self.get_bwt(),
            'FWT': self.get_fwt(),
            'PS': self.get_ps()
        }


def evaluate_task(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
    task_id: Optional[int] = None,
    task_classes: Optional[List[int]] = None
) -> float:
    """
    Evaluate model accuracy on a specific task.

    Args:
        model: PyTorch model
        dataloader: DataLoader for the task
        device: Device to use
        task_id: Task identifier (recorded for logging)
        task_classes: Class indices belonging to this task. When provided, the
            logits are restricted to these classes before the argmax, which is
            the Task-IL protocol (the task identity is known at test time).
            When None, the argmax runs over all classes, which is Class-IL.

    Returns:
        Accuracy on the task
    """
    model.eval()
    correct = 0
    total = 0

    class_index = None
    if task_classes is not None and len(task_classes) > 0:
        class_index = torch.as_tensor(sorted(task_classes),
                                      dtype=torch.long, device=device)

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            if class_index is not None:
                # Restrict to the task's own classes, then map the argmax back
                # to the global label space so it can be compared to targets.
                restricted = outputs.index_select(1, class_index)
                local_pred = restricted.argmax(dim=1)
                predicted = class_index[local_pred]
            else:
                _, predicted = outputs.max(1)

            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    return correct / total if total > 0 else 0.0


def evaluate_all_tasks(
    model: nn.Module,
    dataset,
    tasks_seen: List[int],
    device: str = 'cuda',
    batch_size: int = 64,
    scenario: str = 'class'
) -> Dict[int, float]:
    """
    Evaluate model on all seen tasks.
    
    Args:
        model: PyTorch model
        dataset: Continual dataset instance
        tasks_seen: List of task IDs that have been seen
        device: Device to use
        batch_size: Batch size for evaluation
        
    Returns:
        Dictionary mapping task_id to accuracy
    """
    accuracies = {}

    for task_id in tasks_seen:
        _, _, test_loader = dataset.get_task_loaders(task_id, batch_size=batch_size)

        # Task-IL assumes the task identity is known at test time, so the
        # prediction is taken over that task's classes only. Class-IL takes the
        # argmax over the full label space.
        task_classes = None
        if scenario == 'task':
            task_classes = getattr(dataset, 'task_classes', [None] * (task_id + 1))[task_id]

        acc = evaluate_task(model, test_loader, device, task_id, task_classes)
        accuracies[task_id] = acc

    return accuracies


class MembershipInferenceAttack:
    """
    Membership Inference Attack (MIA) for evaluating unlearning effectiveness.
    
    Tests whether an attacker can determine if a sample was in the training set
    by observing model outputs.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    ):
        """
        Initialize MIA evaluator.
        
        Args:
            model: Target model to attack
            device: Device to use
        """
        self.model = model
        self.device = device
        
    def _compute_confidence(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> np.ndarray:
        """Compute prediction confidence for inputs."""
        self.model.eval()
        
        with torch.no_grad():
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            probs = F.softmax(outputs, dim=1)
            
            # Get confidence for true class
            confidences = probs.gather(1, targets.view(-1, 1)).squeeze()
            
        return confidences.cpu().numpy()
    
    def _compute_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> np.ndarray:
        """Compute loss values for inputs."""
        self.model.eval()
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        with torch.no_grad():
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            losses = criterion(outputs, targets)
            
        return losses.cpu().numpy()
    
    def evaluate(
        self,
        member_loader: DataLoader,
        non_member_loader: DataLoader,
        method: str = 'confidence'
    ) -> Dict[str, float]:
        """
        Evaluate membership inference attack success.
        
        Args:
            member_loader: DataLoader for training (member) samples
            non_member_loader: DataLoader for test (non-member) samples
            method: Attack method ('confidence' or 'loss')
            
        Returns:
            Dictionary with attack metrics (accuracy, AUC, etc.)
        """
        member_scores = []
        non_member_scores = []
        
        # Collect scores for members
        for inputs, targets in member_loader:
            if method == 'confidence':
                scores = self._compute_confidence(inputs, targets)
            else:
                scores = -self._compute_loss(inputs, targets)  # Negate so higher = more member-like
            member_scores.extend(scores)
            
        # Collect scores for non-members
        for inputs, targets in non_member_loader:
            if method == 'confidence':
                scores = self._compute_confidence(inputs, targets)
            else:
                scores = -self._compute_loss(inputs, targets)
            non_member_scores.extend(scores)
            
        member_scores = np.array(member_scores)
        non_member_scores = np.array(non_member_scores)
        
        # Compute attack success rate using threshold-based attack
        all_scores = np.concatenate([member_scores, non_member_scores])
        all_labels = np.concatenate([
            np.ones(len(member_scores)),
            np.zeros(len(non_member_scores))
        ])
        
        # Find optimal threshold
        best_acc = 0.0
        best_threshold = 0.0
        
        for threshold in np.percentile(all_scores, range(0, 101, 5)):
            predictions = (all_scores >= threshold).astype(int)
            acc = accuracy_score(all_labels, predictions)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
                
        # Compute AUC
        try:
            auc = roc_auc_score(all_labels, all_scores)
        except ValueError:
            auc = 0.5
            
        return {
            'mia_accuracy': best_acc * 100,
            'mia_auc': auc * 100,
            'mia_threshold': best_threshold
        }


def compute_kl_divergence(
    model1: nn.Module,
    model2: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda'
) -> float:
    """
    Compute KL divergence between output distributions of two models.
    
    Used to measure how close an unlearned model is to a retrained model.
    
    Args:
        model1: First model (e.g., unlearned model)
        model2: Second model (e.g., retrained model)
        dataloader: DataLoader for evaluation data
        device: Device to use
        
    Returns:
        Average KL divergence
    """
    model1.eval()
    model2.eval()
    
    kl_divs = []
    
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            
            # Get output distributions
            logits1 = model1(inputs)
            logits2 = model2(inputs)
            
            probs1 = F.softmax(logits1, dim=1)
            probs2 = F.softmax(logits2, dim=1)
            
            # Compute KL divergence: KL(P1 || P2)
            kl = F.kl_div(
                probs2.log(),
                probs1,
                reduction='batchmean'
            )
            
            kl_divs.append(kl.item())
            
    return np.mean(kl_divs)


class UnlearningEvaluator:
    """Comprehensive evaluator for machine unlearning."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    ):
        """
        Initialize unlearning evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to use
        """
        self.model = model
        self.device = device
        self.mia = MembershipInferenceAttack(model, device)
        
    def evaluate(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        test_loader: DataLoader,
        retrained_model: Optional[nn.Module] = None
    ) -> Dict[str, float]:
        """
        Evaluate unlearning effectiveness.
        
        Args:
            forget_loader: DataLoader for forget set
            retain_loader: DataLoader for retain set
            test_loader: DataLoader for test set
            retrained_model: Gold-standard retrained model (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = {}
        
        # Accuracy on forget set (should be low/zero)
        results['forget_acc'] = evaluate_task(self.model, forget_loader, self.device) * 100
        
        # Accuracy on retain set (should remain high)
        results['retain_acc'] = evaluate_task(self.model, retain_loader, self.device) * 100
        
        # Overall test accuracy
        results['test_acc'] = evaluate_task(self.model, test_loader, self.device) * 100
        
        # MIA evaluation
        mia_results = self.mia.evaluate(
            forget_loader,
            test_loader,
            method='confidence'
        )
        results['mia'] = mia_results['mia_accuracy']
        results['mia_auc'] = mia_results['mia_auc']
        
        # KL divergence from retrained model
        if retrained_model is not None:
            results['kl_divergence'] = compute_kl_divergence(
                self.model,
                retrained_model,
                test_loader,
                self.device
            )
        else:
            results['kl_divergence'] = None
            
        return results


def measure_execution_time(
    func,
    *args,
    **kwargs
) -> Tuple[float, any]:
    """
    Measure execution time of a function.
    
    Args:
        func: Function to time
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Tuple of (execution_time, function_result)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    return end_time - start_time, result


class ContinualLearningEvaluator:
    """Complete evaluator for continual learning experiments."""
    
    def __init__(
        self,
        model: nn.Module,
        dataset,
        num_tasks: int,
        device: str = 'cuda',
        scenario: str = None
    ):
        """
        Initialize continual learning evaluator.

        Args:
            model: Model to evaluate
            dataset: Continual learning dataset
            num_tasks: Total number of tasks
            device: Device to use
            scenario: 'task' or 'class'. Controls whether predictions are
                restricted to the evaluated task's classes (Task-IL) or taken
                over the full label space (Class-IL). If None, it is read from
                the dataset, defaulting to 'class'.
        """
        self.model = model
        self.dataset = dataset
        self.num_tasks = num_tasks
        self.device = device
        self.scenario = scenario or getattr(dataset, 'scenario', 'class')
        
        self.accuracy_matrix = AccuracyMatrix(num_tasks)
        
        # Set random accuracy for each task (1 / num_classes_in_task)
        for task_id in range(num_tasks):
            classes_in_task = len(dataset.task_classes[task_id])
            self.accuracy_matrix.set_random_accuracy(task_id, 1.0 / classes_in_task)
            
    def evaluate_after_task(
        self,
        current_task: int,
        batch_size: int = 64
    ) -> Dict[int, float]:
        """
        Evaluate model after completing a task.
        
        Args:
            current_task: Task that was just completed
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of task accuracies
        """
        tasks_seen = list(range(current_task + 1))
        accuracies = evaluate_all_tasks(
            self.model,
            self.dataset,
            tasks_seen,
            self.device,
            batch_size,
            scenario=self.scenario
        )
        
        self.accuracy_matrix.update(current_task, accuracies)
        
        return accuracies

    def evaluate_before_task(
        self,
        task_id: int,
        batch_size: int = 64
    ) -> float:
        """
        Evaluate the model on task `task_id` BEFORE training on it.

        Populates A[task_id - 1, task_id], which FWT and the plasticity term of
        PS both require. Call this at the start of each task t >= 1.
        """
        if task_id <= 0:
            return 0.0

        accuracies = evaluate_all_tasks(
            self.model,
            self.dataset,
            [task_id],
            self.device,
            batch_size,
            scenario=self.scenario
        )
        acc = accuracies[task_id]
        self.accuracy_matrix.record_pre_task_accuracy(task_id, acc)

        return acc
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all continual learning metrics."""
        return self.accuracy_matrix.get_all_metrics()
    
    def get_accuracy_matrix(self) -> np.ndarray:
        """Get the full accuracy matrix."""
        return self.accuracy_matrix.matrix.copy()
