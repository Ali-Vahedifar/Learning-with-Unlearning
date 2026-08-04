"""
Training Utilities for LwU Framework

Implements training loops, early stopping, and utilities for both
continual learning and machine unlearning experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from tqdm import tqdm
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min'
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' for loss or accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
                
        return False


class LwUTrainer:
    """Trainer for LwU (Learning with Unlearning) framework."""
    
    def __init__(
        self,
        model,  # LwU model
        device: str = 'cuda',
        lr: float = 0.001,
        weight_decay: float = 0.0,
        patience: int = 10
    ):
        """
        Initialize LwU trainer.
        
        Args:
            model: LwU model instance
            device: Device to use
            lr: Learning rate
            weight_decay: L2 regularization
            patience: Early stopping patience
        """
        self.model = model
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        
        self.criterion = nn.CrossEntropyLoss()
        
    def train_task(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 200,
        task_id: int = 0
    ) -> Dict[str, List[float]]:
        """
        Train on a single task.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum training epochs
            task_id: Current task ID
            
        Returns:
            Dictionary with training history
        """
        optimizer = torch.optim.Adam(
            self.model.backbone.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        early_stopping = EarlyStopping(patience=self.patience, mode='min')
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        self.model.backbone.train()
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model.backbone(inputs)
                loss = self.criterion(outputs, targets)
                
                # Use masked backward to protect Zone A parameters
                self.model.masked_backward(loss)
                
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                train_correct += predicted.eq(targets).sum().item()
                train_total += targets.size(0)
                
            train_loss /= train_total
            train_acc = train_correct / train_total
            
            # Validation phase
            val_loss, val_acc = self._validate(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Task {task_id}, Epoch {epoch+1}/{num_epochs}: "
                    f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                    f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
                )
                
            if early_stopping(val_loss):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
                
        return history
    
    def _validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate model on given data."""
        self.model.backbone.eval()
        
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model.backbone(inputs)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
                
        self.model.backbone.train()
        
        return val_loss / total, correct / total
    
    def continual_learning(
        self,
        dataset,
        num_tasks: int,
        num_epochs: int = 200,
        batch_size: int = 64
    ) -> Dict:
        """
        Run full continual learning experiment.
        
        Args:
            dataset: Continual learning dataset
            num_tasks: Number of tasks
            num_epochs: Epochs per task
            batch_size: Batch size
            
        Returns:
            Experiment results
        """
        from lwu.evaluation.metrics import ContinualLearningEvaluator
        
        evaluator = ContinualLearningEvaluator(
            self.model.backbone,
            dataset,
            num_tasks,
            self.device
        )
        
        all_history = []
        
        for task_id in range(num_tasks):
            logger.info(f"\n{'='*50}")
            logger.info(f"Starting Task {task_id + 1}/{num_tasks}")
            logger.info(f"{'='*50}")
            
            # Record accuracy on this task BEFORE training on it. This fills
            # A[t-1, t], which both FWT and the plasticity term of PS require
            # and which cannot be recovered afterwards.
            if task_id > 0:
                pre_acc = evaluator.evaluate_before_task(task_id, batch_size)
                logger.info(f"Pre-task accuracy on task {task_id + 1}: {pre_acc:.4f}")

            # Get task data
            train_loader, val_loader, _ = dataset.get_task_loaders(
                task_id, batch_size=batch_size
            )
            
            # Prepare for new task (identify zones, apply operations)
            if task_id > 0:
                # Use previous task's data as retain set
                prev_loader = dataset.get_all_seen_loader(
                    list(range(task_id)),
                    batch_size=batch_size,
                    train=True
                )
                self.model.new_task(prev_loader)
                
            # Train on current task
            history = self.train_task(
                train_loader,
                val_loader,
                num_epochs=num_epochs,
                task_id=task_id
            )
            all_history.append(history)
            
            # Evaluate on all seen tasks
            accuracies = evaluator.evaluate_after_task(task_id, batch_size)
            
            logger.info(f"Task {task_id + 1} Accuracies: {accuracies}")
            
            # Log zone statistics
            zone_stats = self.model.get_zone_statistics()
            logger.info(f"Zone Distribution: {zone_stats['overall']}")
            
        # Get final metrics
        final_metrics = evaluator.get_metrics()
        logger.info(f"\nFinal Metrics: {final_metrics}")
        
        return {
            'history': all_history,
            'metrics': final_metrics,
            'accuracy_matrix': evaluator.get_accuracy_matrix()
        }


class BaselineTrainer:
    """Trainer for baseline continual learning methods."""
    
    def __init__(
        self,
        model: nn.Module,
        method,  # Continual learning method (EWC, SI, LwF, etc.)
        device: str = 'cuda',
        lr: float = 0.001,
        weight_decay: float = 0.0,
        patience: int = 10
    ):
        """
        Initialize baseline trainer.
        
        Args:
            model: Neural network model
            method: Continual learning method instance
            device: Device to use
            lr: Learning rate
            weight_decay: L2 regularization
            patience: Early stopping patience
        """
        self.model = model
        self.method = method
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        
        self.criterion = nn.CrossEntropyLoss()
        
    def train_task(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 200,
        task_id: int = 0
    ) -> Dict[str, List[float]]:
        """Train on a single task with regularization."""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        early_stopping = EarlyStopping(patience=self.patience, mode='min')
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        self.model.train()
        
        # For LwF, store teacher before training
        method_name = type(self.method).__name__
        if method_name == 'LwF' and task_id > 0:
            # Get number of classes seen so far
            # This is a simplification - real implementation would track classes
            pass
        
        for epoch in range(num_epochs):
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                
                # Classification loss
                cls_loss = self.criterion(outputs, targets)
                
                # Add method-specific regularization
                if hasattr(self.method, 'penalty'):
                    reg_loss = self.method.penalty()
                    loss = cls_loss + reg_loss
                elif hasattr(self.method, 'combined_loss'):
                    loss = self.method.combined_loss(
                        inputs, targets, outputs, self.criterion
                    )
                else:
                    loss = cls_loss
                    
                loss.backward()
                
                # For SI, accumulate gradients
                if method_name == 'SI':
                    self.method.accumulate_grad()
                    
                optimizer.step()
                
                train_loss += cls_loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                train_correct += predicted.eq(targets).sum().item()
                train_total += targets.size(0)
                
            train_loss /= train_total
            train_acc = train_correct / train_total
            
            val_loss, val_acc = self._validate(val_loader)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Task {task_id}, Epoch {epoch+1}/{num_epochs}: "
                    f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                    f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
                )
                
            if early_stopping(val_loss):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
                
        # Post-task consolidation
        method_name = type(self.method).__name__
        if method_name == 'EWC':
            self.method.consolidate(train_loader)
        elif method_name == 'SI':
            self.method.update_omega()
        elif method_name == 'LwF':
            # Store current model as teacher for next task
            num_classes = outputs.size(1)
            self.method.store_teacher(num_classes)
            
        return history
    
    def _validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate model on given data."""
        self.model.eval()
        
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
                
        self.model.train()
        
        return val_loss / total, correct / total
    
    def continual_learning(
        self,
        dataset,
        num_tasks: int,
        num_epochs: int = 200,
        batch_size: int = 64
    ) -> Dict:
        """Run full continual learning experiment."""
        from lwu.evaluation.metrics import ContinualLearningEvaluator
        
        evaluator = ContinualLearningEvaluator(
            self.model,
            dataset,
            num_tasks,
            self.device
        )
        
        all_history = []
        
        for task_id in range(num_tasks):
            logger.info(f"\n{'='*50}")
            logger.info(f"Starting Task {task_id + 1}/{num_tasks}")
            logger.info(f"{'='*50}")

            if task_id > 0:
                pre_acc = evaluator.evaluate_before_task(task_id, batch_size)
                logger.info(f"Pre-task accuracy on task {task_id + 1}: {pre_acc:.4f}")

            train_loader, val_loader, _ = dataset.get_task_loaders(
                task_id, batch_size=batch_size
            )
            
            history = self.train_task(
                train_loader,
                val_loader,
                num_epochs=num_epochs,
                task_id=task_id
            )
            all_history.append(history)
            
            accuracies = evaluator.evaluate_after_task(task_id, batch_size)
            logger.info(f"Task {task_id + 1} Accuracies: {accuracies}")
            
        final_metrics = evaluator.get_metrics()
        logger.info(f"\nFinal Metrics: {final_metrics}")
        
        return {
            'history': all_history,
            'metrics': final_metrics,
            'accuracy_matrix': evaluator.get_accuracy_matrix()
        }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = 'cuda'
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Args:
        model: Neural network model
        dataloader: Training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to use
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
        
    return total_loss / total, correct / total


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = 'cuda'
) -> Tuple[float, float]:
    """
    Evaluate model.
    
    Args:
        model: Neural network model
        dataloader: Evaluation data
        criterion: Loss function
        device: Device to use
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
            
    return total_loss / total, correct / total


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
