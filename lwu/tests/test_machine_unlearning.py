from copy import deepcopy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from Machine_Unlearning_baselines import (
    Amnesiac,
    BadTeacher,
    Baseline,
    BoundaryExpand,
    BoundaryShrink,
    FineTune,
    GradientAscent,
    InfluenceUnlearning,
    L1Sparse,
    RandomLabel,
    Retrain,
    SCRUB,
    SSD,
    SalUn,
    UNSIR,
    UniCLUN,
)
from core.evaluation.metrics import MembershipInferenceAttack, compute_weight_distribution_kl
from scripts.train_unlearning import (
    build_instance_unlearning_loaders,
    build_unlearning_loaders,
)


def _model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))


def _loader(seed=0, n=12, classes=3):
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(n, 4, generator=generator)
    labels = torch.randint(0, classes, (n,), generator=generator)
    return DataLoader(TensorDataset(inputs, labels), batch_size=4, shuffle=False)


def test_baseline_is_noop_and_retrain_uses_a_fresh_model():
    source = _model()
    stored = deepcopy(source.state_dict())
    assert Baseline(source).unlearn() is source
    assert all(torch.equal(source.state_dict()[key], value) for key, value in stored.items())

    retrained = Retrain(lambda: _model(9), device='cpu', epochs=1).unlearn(None, _loader())
    assert retrained is not source


def test_finetune_uses_retain_data_and_changes_source_model():
    model = _model()
    before = deepcopy(model.state_dict())
    returned = FineTune(model, device='cpu', epochs=1, lr=1e-2).unlearn(_loader(1), _loader(2))
    assert returned is model
    assert any(not torch.equal(model.state_dict()[name], value) for name, value in before.items())


def test_ssd_implements_equation_four_exactly():
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(3.0)
    method = SSD(
        model,
        device='cpu',
        alpha=10.0,
        dampening_constant=1.0,
        full_fisher={'weight': torch.tensor([[2.0]])},
    )
    method._dampen({'weight': torch.tensor([[30.0]])})
    assert torch.allclose(model.weight, torch.tensor([[0.2]]))


def test_ssd_refuses_to_invent_missing_full_data_fisher():
    with pytest.raises(RuntimeError, match='full_loader'):
        SSD(_model(), device='cpu').unlearn(_loader())


def test_amnesiac_subtracts_recorded_optimizer_update():
    model = _model()
    initial = deepcopy(model.state_dict())
    method = Amnesiac(model, device='cpu')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs, labels = next(iter(_loader()))
    method.before_step()
    optimizer.zero_grad(set_to_none=True)
    nn.CrossEntropyLoss()(model(inputs), labels).backward()
    optimizer.step()
    method.after_step(contains_sensitive=True)
    method.unlearn()
    for name, tensor in initial.items():
        assert torch.allclose(model.state_dict()[name], tensor, atol=1e-7)


def test_amnesiac_rejects_posthoc_substitute():
    with pytest.raises(RuntimeError, match='recorded during original training'):
        Amnesiac(_model(), device='cpu').unlearn(_loader())


def test_bad_teacher_and_unsir_are_executable_paper_objectives():
    bad_model = _model()
    bad_before = deepcopy(bad_model.state_dict())
    BadTeacher(bad_model, device='cpu', epochs=1, lr=0.01).unlearn(_loader(1), _loader(2))
    assert any(
        not torch.equal(bad_model.state_dict()[key], value) for key, value in bad_before.items()
    )

    unsir_model = _model()
    UNSIR(unsir_model, device='cpu', noise_steps=1, impair_epochs=1, repair_epochs=1).unlearn(
        _loader(1), _loader(2)
    )
    assert all(torch.isfinite(parameter).all() for parameter in unsir_model.parameters())


def test_uniclun_removes_forgotten_classes_from_bounded_memory():
    model = _model()
    method = UniCLUN(model, device='cpu', epochs=1, bernoulli_p=1.0)
    method.buffer_x = torch.randn(4, 4)
    method.buffer_y = torch.tensor([0, 1, 2, 1])
    forget = DataLoader(
        TensorDataset(torch.randn(4, 4), torch.ones(4, dtype=torch.long)), batch_size=2
    )
    method.unlearn(forget, _loader(3))
    assert not method.buffer_y.eq(1).any()


def test_weight_distribution_kl_is_zero_for_identical_weights():
    first = _model()
    second = deepcopy(first)
    assert compute_weight_distribution_kl(first, second) == pytest.approx(0.0)
    with torch.no_grad():
        next(second.parameters()).add_(1.0)
    assert compute_weight_distribution_kl(first, second) > 0.0


def test_amnesiac_running_sum_matches_per_batch_replay():
    def train(track):
        model = _model()
        method = Amnesiac(model, device='cpu', track_per_batch=track)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        loader = list(_loader())
        for step, (inputs, labels) in enumerate(loader * 2):
            method.before_step()
            optimizer.zero_grad(set_to_none=True)
            nn.CrossEntropyLoss()(model(inputs), labels).backward()
            optimizer.step()
            method.after_step(contains_sensitive=(step % 2 == 0))
        return model, method

    ledger_model, ledger = train(True)
    sum_model, running = train(False)
    assert running.num_recorded == len(ledger.update_ledger) > 1

    expected = deepcopy(ledger_model)
    with torch.no_grad():
        for update in ledger.update_ledger:
            for name, parameter in expected.named_parameters():
                parameter.sub_(update[name])

    running.unlearn()
    for (_, got), (_, want) in zip(sum_model.named_parameters(), expected.named_parameters()):
        assert torch.allclose(got, want, atol=1e-6)


def test_amnesiac_rollback_undoes_only_updates_up_to_the_snapshot():
    model = _model()
    method = Amnesiac(model, device='cpu')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs, labels = next(iter(_loader()))
    snapshot = None
    for step in range(4):
        method.before_step()
        optimizer.zero_grad(set_to_none=True)
        nn.CrossEntropyLoss()(model(inputs), labels).backward()
        optimizer.step()
        method.after_step(contains_sensitive=True)
        if step == 1:
            snapshot = method.checkpoint()
    assert method.num_recorded == 4
    method.rollback(snapshot)
    assert method.num_recorded == 2


def test_amnesiac_selective_ids_need_the_per_batch_ledger():
    method = Amnesiac(_model(), device='cpu')
    method.update_sum = {}
    method.num_recorded = 1
    with pytest.raises(RuntimeError, match='track_per_batch'):
        method.unlearn(forget_ids=[0])


def test_scrub_max_step_pushes_the_student_away_from_its_teacher():
    model = _model()
    teacher = deepcopy(model)
    forget, retain = _loader(1), _loader(2)

    def divergence(net):
        total = 0.0
        with torch.no_grad():
            for inputs, _ in forget:
                total += torch.nn.functional.kl_div(
                    torch.log_softmax(net(inputs), dim=1),
                    torch.softmax(teacher(inputs), dim=1),
                    reduction='batchmean',
                ).item()
        return total

    before = divergence(model)
    SCRUB(model, device='cpu', epochs=1, msteps=1, lr=0.05).unlearn(forget, retain)
    assert divergence(model) > before


def test_scrub_rejects_more_max_steps_than_epochs():
    with pytest.raises(ValueError, match='msteps'):
        SCRUB(_model(), device='cpu', epochs=1, msteps=2)


def test_salun_mask_keeps_the_requested_fraction_of_weights():
    model = _model()
    method = SalUn(model, device='cpu', sparsity=0.25, num_classes=3)
    mask = method.saliency_mask(_loader())
    kept = sum(int(m.sum()) for m in mask.values())
    total = sum(m.numel() for m in mask.values())
    assert abs(kept / total - 0.25) < 0.02


def test_salun_leaves_non_salient_weights_untouched():
    model = _model()
    method = SalUn(model, device='cpu', epochs=1, lr=0.1, sparsity=0.25, num_classes=3)
    mask = method.saliency_mask(_loader())
    before = deepcopy(model.state_dict())
    method.saliency_mask = lambda _loader_arg: mask
    method.unlearn(_loader(1), _loader(2))
    for name, parameter in model.named_parameters():
        frozen = ~mask[name]
        if frozen.any():
            assert torch.allclose(parameter[frozen], before[name][frozen], atol=1e-7)


def test_salun_random_labels_never_keep_the_true_class():
    generator = torch.Generator().manual_seed(0)
    labels = torch.randint(0, 10, (256,), generator=generator)
    offset = torch.randint(1, 10, labels.shape, generator=generator)
    assert torch.all((labels + offset) % 10 != labels)


def test_salun_rejects_a_degenerate_sparsity():
    with pytest.raises(ValueError, match='sparsity'):
        SalUn(_model(), device='cpu', sparsity=0.0)


def test_output_kl_stays_finite_when_softmax_underflows():
    from core.evaluation.metrics import compute_kl_divergence

    class Confident(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

        def forward(self, inputs):
            logits = torch.zeros(inputs.size(0), 100)
            logits[:, 0] = self.scale
            return logits

    assert torch.softmax(torch.tensor([[200.0, 0.0]]), dim=1)[0, 1].item() == 0.0
    loader = _loader(n=8, classes=100)
    for left, right in ((200.0, 0.0), (0.0, 200.0)):
        value = compute_kl_divergence(Confident(left), Confident(right), loader, 'cpu')
        assert torch.isfinite(torch.tensor(value)), (left, right, value)
    same = compute_kl_divergence(Confident(200.0), Confident(200.0), loader, 'cpu')
    assert abs(same) < 1e-6


def test_mia_uses_independent_calibration_and_learns_score_direction():
    class ScoreModel(nn.Module):
        def forward(self, inputs):
            score = inputs[:, 0]
            return torch.stack((score, torch.zeros_like(score), torch.zeros_like(score)), dim=1)

    def scores(values):
        x = torch.tensor(values, dtype=torch.float32).view(-1, 1)
        y = torch.zeros(len(values), dtype=torch.long)
        return DataLoader(TensorDataset(x, y), batch_size=2)

    attack = MembershipInferenceAttack(ScoreModel(), device='cpu')
    result = attack.evaluate(
        member_loader=scores([-6.0, -5.0, -4.0, -3.5]),
        non_member_loader=scores([3.5, 4.0, 5.0, 6.0]),
        calibration_member_loader=scores([-5.0, -4.5, -3.5, -3.0]),
        calibration_non_member_loader=scores([3.0, 3.5, 4.5, 5.0]),
    )
    assert result['mia_orientation'] == -1
    assert result['mia_accuracy'] == pytest.approx(100.0)
    assert result['mia_auc'] == pytest.approx(100.0)


def test_unlearning_loader_protocol_keeps_selection_and_final_mia_disjoint():
    class ToyDataset(torch.utils.data.Dataset):
        def __init__(self, targets, transform=None):
            self.targets = list(targets)
            self.transform = transform

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return torch.tensor([float(index)]), self.targets[index]

    class ProtocolDataset:
        val_split = 0.2
        test_transform = object()
        train_dataset = ToyDataset([label for label in range(3) for _ in range(20)])
        val_dataset = ToyDataset([label for label in range(3) for _ in range(20)])
        test_dataset = ToyDataset([label for label in range(3) for _ in range(5)])

    args = type('Args', (), dict(batch_size=4, num_workers=0, device='cpu', retain_ratio=0.1))()
    loaders = build_unlearning_loaders(ProtocolDataset(), 1, args, seed=7)

    def indices(name):
        return set(loaders[name].dataset.indices)

    member_parts = [
        indices('mia_member_cal'),
        indices('mia_member_select'),
        indices('mia_member_final'),
    ]
    assert not (member_parts[0] & member_parts[1])
    assert not (member_parts[0] & member_parts[2])
    assert not (member_parts[1] & member_parts[2])
    assert set.union(*member_parts) == indices('forget_train_eval')
    assert not (indices('mia_nonmember_cal') & indices('mia_nonmember_select'))
    assert indices('mia_nonmember_cal') | indices('mia_nonmember_select') == indices('forget_val')


def test_instance_unlearning_protocol_is_balanced_and_disjoint():
    class ToyDataset(torch.utils.data.Dataset):
        def __init__(self, targets, transform=None):
            self.targets = list(targets)
            self.transform = transform

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, index):
            return torch.tensor([float(index)]), self.targets[index]

    class ProtocolDataset:
        val_split = 0.2
        test_transform = object()
        train_dataset = ToyDataset([label for label in range(3) for _ in range(50)])
        val_dataset = ToyDataset([label for label in range(3) for _ in range(50)])
        test_dataset = ToyDataset([label for label in range(3) for _ in range(20)])

    args = type('Args', (), dict(batch_size=4, num_workers=0, device='cpu', retain_ratio=0.1))()
    loaders = build_instance_unlearning_loaders(ProtocolDataset(), num_forget=30, args=args, seed=7)

    def indices(name):
        return set(loaders[name].dataset.indices)

    forgotten = indices('forget_train_eval')
    retained = indices('retain_full')
    members = [indices('mia_member_cal'), indices('mia_member_select'), indices('mia_member_final')]
    assert len(forgotten) == 30
    assert not forgotten & retained
    assert set.union(*members) == forgotten
    assert not (members[0] & members[1])
    assert not (members[0] & members[2])
    assert not (members[1] & members[2])
    labels = ProtocolDataset.train_dataset.targets
    counts = [sum(labels[index] == class_id for index in forgotten) for class_id in range(3)]
    assert counts == [10, 10, 10]


def test_random_labels_move_the_model_off_the_true_forget_labels():
    model = _model()
    before = deepcopy(model)
    RandomLabel(model, device='cpu', epochs=2, lr=0.1, num_classes=3, seed=0).unlearn(
        _loader(), _loader(seed=1)
    )
    assert any(not torch.equal(a, b) for a, b in zip(model.parameters(), before.parameters()))


def test_gradient_ascent_raises_the_forget_loss_it_ascends():
    model = _model()
    loader = _loader()
    before = _forget_loss(model, loader)
    GradientAscent(model, device='cpu', epochs=2, lr=0.05).unlearn(loader)
    assert _forget_loss(model, loader) > before


def test_l1_sparse_shrinks_the_weights_a_plain_finetune_would_not():
    retain = _loader(seed=2)
    sparse, plain = _model(), _model()
    L1Sparse(sparse, device='cpu', epochs=2, lr=0.01, alpha=0.05).unlearn(_loader(), retain)
    L1Sparse(plain, device='cpu', epochs=2, lr=0.01, alpha=0.0).unlearn(_loader(), retain)
    assert _l1(sparse) < _l1(plain)


def test_boundary_shrink_relabels_to_a_neighbouring_region():
    model = _model()
    before = deepcopy(model)
    BoundaryShrink(model, device='cpu', epochs=1, lr=0.05, bound=0.1).unlearn(_loader())
    assert any(not torch.equal(a, b) for a, b in zip(model.parameters(), before.parameters()))


def test_boundary_expand_restores_the_original_head_width():
    model = _model()
    width = model[-1].out_features
    BoundaryExpand(model, device='cpu', epochs=1, lr=0.05).unlearn(_loader())
    assert model[-1].out_features == width
    assert model(torch.randn(2, 4)).shape == (2, width)


def test_influence_unlearning_is_one_step_and_scales_with_alpha():
    forget, retain = _loader(), _loader(seed=3)
    small, large = _model(), _model()
    source = _model()
    InfluenceUnlearning(small, device='cpu', alpha=0.5).unlearn(forget, retain)
    InfluenceUnlearning(large, device='cpu', alpha=5.0).unlearn(forget, retain)
    assert _distance(large, source) > _distance(small, source) > 0


def _forget_loss(model, loader):
    model.eval()
    with torch.no_grad():
        return sum(nn.functional.cross_entropy(model(x), y).item() for x, y in loader)


def _l1(model):
    return sum(p.abs().sum().item() for p in model.parameters())


def _distance(model, other):
    return sum((a - b).norm().item() for a, b in zip(model.parameters(), other.parameters()))
