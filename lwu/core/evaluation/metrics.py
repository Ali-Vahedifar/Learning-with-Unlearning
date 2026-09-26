from contextlib import nullcontext
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader


def evaluate_task(
    model: nn.Module, dataloader: DataLoader, device: str = 'cuda', test_time_adapt: bool = False
) -> float:
    """Accuracy on a loader.

    `test_time_adapt` is for methods that keep an inference-time memory (LwU):
    the stream context bounds the memory's lifetime to this evaluation, and each
    batch is predicted before it is written, so the memory can only help later
    batches.
    """
    model.eval()
    correct = total = 0
    stream = (
        model.test_stream() if test_time_adapt and hasattr(model, 'test_stream') else nullcontext()
    )
    with stream, torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, test_time_adapt=True) if test_time_adapt else model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    return correct / total if total > 0 else 0.0


class MembershipInferenceAttack:
    def __init__(self, model: nn.Module, device: str = 'cuda', test_time_adapt: bool = False):
        self.model = model
        self.device = device
        self.test_time_adapt = test_time_adapt

    def _forward(self, inputs):
        if self.test_time_adapt:
            return self.model(inputs, test_time_adapt=True)
        return self.model(inputs)

    def _compute_confidence(self, inputs, targets):
        self.model.eval()
        with torch.no_grad():
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            probs = F.softmax(self._forward(inputs), dim=1)
            confidences = probs.gather(1, targets.view(-1, 1)).squeeze()
        return confidences.cpu().numpy()

    def _compute_loss(self, inputs, targets):
        self.model.eval()
        criterion = nn.CrossEntropyLoss(reduction='none')
        with torch.no_grad():
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            losses = criterion(self._forward(inputs), targets)
        return losses.cpu().numpy()

    def evaluate(
        self,
        member_loader: DataLoader,
        non_member_loader: DataLoader,
        method: str = 'confidence',
        calibration_member_loader: Optional[DataLoader] = None,
        calibration_non_member_loader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        def collect(loader):
            collected = []
            stream = (
                self.model.test_stream()
                if self.test_time_adapt and hasattr(self.model, 'test_stream')
                else nullcontext()
            )
            with stream:
                for inputs, targets in loader:
                    if method == 'confidence':
                        scores = self._compute_confidence(inputs, targets)
                    else:
                        scores = -self._compute_loss(inputs, targets)
                    collected.extend(scores)
            return np.asarray(collected)

        member_scores = collect(member_loader)
        non_member_scores = collect(non_member_loader)
        if len(member_scores) == 0 or len(non_member_scores) == 0:
            raise ValueError('MIA requires non-empty member and non-member sets')
        if (calibration_member_loader is None) != (calibration_non_member_loader is None):
            raise ValueError('both MIA calibration loaders must be provided together')

        if calibration_member_loader is not None:
            cal_member = collect(calibration_member_loader)
            cal_nonmember = collect(calibration_non_member_loader)
            eval_member, eval_nonmember = member_scores, non_member_scores
        else:
            rng = np.random.default_rng(0)
            member_scores = member_scores[rng.permutation(len(member_scores))]
            non_member_scores = non_member_scores[rng.permutation(len(non_member_scores))]
            m_cut = max(1, len(member_scores) // 2)
            n_cut = max(1, len(non_member_scores) // 2)
            cal_member, cal_nonmember = member_scores[:m_cut], non_member_scores[:n_cut]
            eval_member = member_scores[m_cut:] if m_cut < len(member_scores) else member_scores
            eval_nonmember = (
                non_member_scores[n_cut:] if n_cut < len(non_member_scores) else non_member_scores
            )

        if len(cal_member) == 0 or len(cal_nonmember) == 0:
            raise ValueError('MIA calibration requires both member and non-member data')
        cal_scores = np.concatenate([cal_member, cal_nonmember])
        cal_labels = np.concatenate([np.ones(len(cal_member)), np.zeros(len(cal_nonmember))])
        eval_scores = np.concatenate([eval_member, eval_nonmember])
        eval_labels = np.concatenate([np.ones(len(eval_member)), np.zeros(len(eval_nonmember))])

        best_threshold = float(np.median(cal_scores))
        best_orientation = 1
        best_cal = -1.0
        for threshold in np.unique(np.percentile(cal_scores, range(0, 101))):
            for orientation in (1, -1):
                predictions = orientation * cal_scores >= orientation * threshold
                score = balanced_accuracy_score(cal_labels, predictions)
                if score > best_cal:
                    best_cal = score
                    best_threshold = float(threshold)
                    best_orientation = orientation

        attack_predictions = best_orientation * eval_scores >= best_orientation * best_threshold
        attack_acc = balanced_accuracy_score(eval_labels, attack_predictions)
        try:
            auc = roc_auc_score(eval_labels, best_orientation * eval_scores)
        except ValueError:
            auc = 0.5
        return {
            'mia_accuracy': attack_acc * 100,
            'mia_auc': auc * 100,
            'mia_threshold': best_threshold,
            'mia_orientation': best_orientation,
            'mia_calibration_accuracy': best_cal * 100,
        }


def compute_kl_divergence(
    model1: nn.Module,
    model2: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
    test_time_adapt: bool = False,
) -> float:
    model1.eval()
    model2.eval()
    kl_divs = []
    stream = (
        model1.test_stream()
        if test_time_adapt and hasattr(model1, 'test_stream')
        else nullcontext()
    )
    with stream, torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            logits1 = model1(inputs, test_time_adapt=True) if test_time_adapt else model1(inputs)
            probs1 = F.softmax(logits1, dim=1)
            log_probs2 = F.log_softmax(model2(inputs), dim=1)
            kl_divs.append(F.kl_div(log_probs2, probs1, reduction='batchmean').item())
    return np.mean(kl_divs)


def compute_weight_distribution_kl(
    model1: nn.Module, model2: nn.Module, bins: int = 256, epsilon: float = 1e-12
) -> float:
    def flat(model):
        return torch.cat([p.detach().float().cpu().reshape(-1) for p in model.parameters()]).numpy()

    weights1, weights2 = flat(model1), flat(model2)
    low = float(min(weights1.min(), weights2.min()))
    high = float(max(weights1.max(), weights2.max()))
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError('model weights contain NaN or infinity')
    if high == low:
        return 0.0
    hist1, edges = np.histogram(weights1, bins=bins, range=(low, high))
    hist2, _ = np.histogram(weights2, bins=edges)
    p = hist1.astype(np.float64) + epsilon
    q = hist2.astype(np.float64) + epsilon
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


class UnlearningEvaluator:
    def __init__(self, model: nn.Module, device: str = 'cuda', test_time_adapt: bool = False):
        self.model = model
        self.device = device
        self.test_time_adapt = test_time_adapt
        self.mia = MembershipInferenceAttack(model, device, test_time_adapt=test_time_adapt)

    def evaluate(
        self,
        forget_loader: DataLoader,
        retain_loader: DataLoader,
        test_loader: DataLoader,
        retrained_model: Optional[nn.Module] = None,
        non_member_loader: Optional[DataLoader] = None,
        forget_eval_loader: Optional[DataLoader] = None,
        retain_eval_loader: Optional[DataLoader] = None,
        mia_calibration_member_loader: Optional[DataLoader] = None,
        mia_calibration_non_member_loader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        adapt = self.test_time_adapt
        results = {
            'forget_acc': evaluate_task(
                self.model, forget_eval_loader or forget_loader, self.device, test_time_adapt=adapt
            )
            * 100,
            'retain_acc': evaluate_task(
                self.model, retain_eval_loader or retain_loader, self.device, test_time_adapt=adapt
            )
            * 100,
            'test_acc': evaluate_task(self.model, test_loader, self.device, test_time_adapt=adapt)
            * 100,
        }
        mia = self.mia.evaluate(
            forget_loader,
            non_member_loader or test_loader,
            method='confidence',
            calibration_member_loader=mia_calibration_member_loader,
            calibration_non_member_loader=mia_calibration_non_member_loader,
        )
        results['mia'] = mia['mia_accuracy']
        results['mia_auc'] = mia['mia_auc']
        if retrained_model is not None:
            results['kl_divergence'] = compute_weight_distribution_kl(self.model, retrained_model)
            results['output_kl_divergence'] = compute_kl_divergence(
                self.model,
                retrained_model,
                test_loader,
                self.device,
                test_time_adapt=self.test_time_adapt,
            )
        else:
            results['kl_divergence'] = None
        return results
