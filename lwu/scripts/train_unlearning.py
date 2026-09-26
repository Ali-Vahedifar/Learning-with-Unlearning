import copy
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Machine_Unlearning_baselines import Amnesiac
from core.utils.training import build_optimizer, evaluate


def _targets(dataset):
    values = dataset.targets
    return np.asarray(values, dtype=np.int64)


def build_unlearning_loaders(dataset, forget_class, args, seed):
    rng = np.random.default_rng(seed)
    clean_train = copy.copy(dataset.train_dataset)
    clean_train.transform = dataset.test_transform
    targets = _targets(dataset.train_dataset)
    train_indices, val_indices = [], []
    for class_id in np.unique(targets):
        indices = np.flatnonzero(targets == class_id)
        rng.shuffle(indices)
        val_n = max(1, int(round(len(indices) * dataset.val_split)))
        val_indices.extend(indices[:val_n].tolist())
        train_indices.extend(indices[val_n:].tolist())
    forget_train = [i for i in train_indices if targets[i] == forget_class]
    retain_train = [i for i in train_indices if targets[i] != forget_class]
    forget_val = [i for i in val_indices if targets[i] == forget_class]
    retain_val = [i for i in val_indices if targets[i] != forget_class]
    if not forget_train or not retain_train:
        raise ValueError(f'forget class {forget_class} does not produce both partitions')
    if not forget_val or not retain_val:
        raise ValueError('validation split does not contain both forget and retain data')
    retain_small_n = max(1, int(round(len(retain_train) * args.retain_ratio)))
    retain_small = rng.choice(retain_train, size=retain_small_n, replace=False).tolist()

    mia_members = np.asarray(forget_train, dtype=np.int64)
    rng.shuffle(mia_members)
    member_cal_end = max(1, len(mia_members) // 2)
    member_select_end = max(
        member_cal_end + 1, member_cal_end + (len(mia_members) - member_cal_end) // 2
    )
    member_cal = mia_members[:member_cal_end].tolist()
    member_select = mia_members[member_cal_end:member_select_end].tolist()
    member_final = mia_members[member_select_end:].tolist()
    if not member_select or not member_final:
        raise ValueError('forget split is too small for independent MIA subsets')

    mia_val_nonmembers = np.asarray(forget_val, dtype=np.int64)
    rng.shuffle(mia_val_nonmembers)
    nonmember_cal_end = max(1, len(mia_val_nonmembers) // 2)
    nonmember_cal = mia_val_nonmembers[:nonmember_cal_end].tolist()
    nonmember_select = mia_val_nonmembers[nonmember_cal_end:].tolist()
    if not nonmember_select:
        raise ValueError('forget validation split is too small for MIA calibration')

    test_targets = _targets(dataset.test_dataset)
    forget_test = np.flatnonzero(test_targets == forget_class).tolist()
    retain_test = np.flatnonzero(test_targets != forget_class).tolist()

    def loader(source, indices, shuffle=False):
        return DataLoader(
            Subset(source, indices),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=str(args.device).startswith('cuda'),
        )

    return {
        'full_train': loader(dataset.train_dataset, train_indices, True),
        'full_val': loader(getattr(dataset, 'val_dataset', dataset.train_dataset), val_indices),
        'full_test': loader(dataset.test_dataset, list(range(len(dataset.test_dataset)))),
        'forget_train': loader(dataset.train_dataset, forget_train, True),
        'forget_train_eval': loader(clean_train, forget_train),
        'retain_full': loader(dataset.train_dataset, retain_train, True),
        'retain_small': loader(dataset.train_dataset, retain_small, True),
        'forget_val': loader(dataset.val_dataset, forget_val),
        'retain_val': loader(dataset.val_dataset, retain_val),
        'mia_member_cal': loader(clean_train, member_cal),
        'mia_member_select': loader(clean_train, member_select),
        'mia_member_final': loader(clean_train, member_final),
        'mia_nonmember_cal': loader(dataset.val_dataset, nonmember_cal),
        'mia_nonmember_select': loader(dataset.val_dataset, nonmember_select),
        'forget_test': loader(dataset.test_dataset, forget_test),
        'retain_test': loader(dataset.test_dataset, retain_test),
    }


def build_instance_unlearning_loaders(dataset, num_forget, args, seed):
    if num_forget < 6:
        raise ValueError('instance unlearning requires at least 6 forget samples')

    rng = np.random.default_rng(seed)
    clean_train = copy.copy(dataset.train_dataset)
    clean_train.transform = dataset.test_transform
    targets = _targets(dataset.train_dataset)
    classes = np.unique(targets)

    train_by_class, val_by_class = {}, {}
    train_indices, val_indices = [], []
    for class_id in classes:
        indices = np.flatnonzero(targets == class_id)
        rng.shuffle(indices)
        val_n = max(1, int(round(len(indices) * dataset.val_split)))
        val = indices[:val_n].tolist()
        train = indices[val_n:].tolist()
        val_by_class[int(class_id)] = val
        train_by_class[int(class_id)] = train
        val_indices.extend(val)
        train_indices.extend(train)

    if num_forget >= len(train_indices):
        raise ValueError('num_forget must be smaller than the training split')

    base, remainder = divmod(num_forget, len(classes))
    class_order = classes.copy()
    rng.shuffle(class_order)
    forget_by_class = {}
    for rank, class_id in enumerate(class_order):
        count = base + int(rank < remainder)
        pool = train_by_class[int(class_id)]
        if count > len(pool):
            raise ValueError('num_forget exceeds the available balanced pool')
        forget_by_class[int(class_id)] = pool[:count]

    member_cal, member_select, member_final = [], [], []
    nonmember_cal, nonmember_select, nonmember_final = [], [], []
    test_targets = _targets(dataset.test_dataset)
    for class_id in classes:
        members = np.asarray(forget_by_class[int(class_id)], dtype=np.int64)
        rng.shuffle(members)
        cal_end = max(1, len(members) // 2)
        select_end = max(cal_end + 1, cal_end + (len(members) - cal_end) // 2)
        parts = (members[:cal_end], members[cal_end:select_end], members[select_end:])
        if any(len(part) == 0 for part in parts):
            raise ValueError(
                'each class needs enough forgotten instances for '
                'calibration, selection, and final subsets'
            )
        member_cal.extend(parts[0].tolist())
        member_select.extend(parts[1].tolist())
        member_final.extend(parts[2].tolist())

        val_pool = np.asarray(val_by_class[int(class_id)], dtype=np.int64)
        rng.shuffle(val_pool)
        cal_n, select_n = len(parts[0]), len(parts[1])
        if cal_n + select_n > len(val_pool):
            raise ValueError('validation split is too small for matched MIA data')
        nonmember_cal.extend(val_pool[:cal_n].tolist())
        nonmember_select.extend(val_pool[cal_n : cal_n + select_n].tolist())

        test_pool = np.flatnonzero(test_targets == class_id)
        rng.shuffle(test_pool)
        if len(parts[2]) > len(test_pool):
            raise ValueError('test split is too small for matched final MIA data')
        nonmember_final.extend(test_pool[: len(parts[2])].tolist())

    forget_train = [index for values in forget_by_class.values() for index in values]
    forget_set = set(forget_train)
    retain_train = [index for index in train_indices if index not in forget_set]
    retain_small_n = max(1, int(round(len(retain_train) * args.retain_ratio)))
    retain_small = rng.choice(retain_train, size=retain_small_n, replace=False).tolist()

    def loader(source, indices, shuffle=False):
        return DataLoader(
            Subset(source, list(indices)),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=str(args.device).startswith('cuda'),
        )

    all_test = list(range(len(dataset.test_dataset)))
    return {
        'full_train': loader(dataset.train_dataset, train_indices, True),
        'full_val': loader(dataset.val_dataset, val_indices),
        'full_test': loader(dataset.test_dataset, all_test),
        'forget_train': loader(dataset.train_dataset, forget_train, True),
        'forget_train_eval': loader(clean_train, forget_train),
        'retain_full': loader(dataset.train_dataset, retain_train, True),
        'retain_small': loader(dataset.train_dataset, retain_small, True),
        'forget_val': loader(clean_train, member_select),
        'retain_val': loader(dataset.val_dataset, val_indices),
        'mia_member_cal': loader(clean_train, member_cal),
        'mia_member_select': loader(clean_train, member_select),
        'mia_member_final': loader(clean_train, member_final),
        'mia_nonmember_cal': loader(dataset.val_dataset, nonmember_cal),
        'mia_nonmember_select': loader(dataset.val_dataset, nonmember_select),
        'mia_nonmember_final': loader(dataset.test_dataset, nonmember_final),
        'forget_test': loader(clean_train, member_final),
        'retain_test': loader(dataset.test_dataset, all_test),
    }


def train_baseline_model(model, train_loader, val_loader, args, forget_class):
    device = args.device
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        getattr(args, 'optimizer', 'adam'),
        model.parameters(),
        args.lr,
        weight_decay=getattr(args, 'weight_decay', 0.0),
        momentum=getattr(args, 'momentum', 0.9),
    )
    scheduler_name = getattr(args, 'scheduler', 'plateau')
    if scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.num_epochs), eta_min=getattr(args, 'min_lr', 0.0)
        )
    elif scheduler_name == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(getattr(args, 'lr_milestones', (60, 120))),
            gamma=getattr(args, 'lr_gamma', 0.2),
        )
    elif scheduler_name == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    else:
        raise ValueError(f'unknown source scheduler {scheduler_name!r}')

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    best_epoch = 0
    epochs_ran = 0
    early_stopping_patience = getattr(args, 'early_stopping_patience', 10)
    min_epochs = getattr(args, 'min_epochs', 0)

    amnesiac = Amnesiac(model, device=device) if args.method == 'amnesiac' else None
    best_amnesiac_state = None

    for epoch in range(args.num_epochs):
        epochs_ran = epoch + 1
        model.train()
        total_loss = correct = total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            if amnesiac is not None:
                amnesiac.before_step()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if amnesiac is not None:
                amnesiac.after_step(contains_sensitive=bool(targets.eq(forget_class).any().item()))
            total_loss += loss.item() * targets.size(0)
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)
        train_loss, train_acc = total_loss / total, correct / total

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        if scheduler_name == 'plateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_amnesiac_state = amnesiac.checkpoint() if amnesiac is not None else None
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{args.num_epochs}: "
                f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                f"Val Acc={val_acc:.4f}"
            )

        if (
            early_stopping_patience > 0
            and epoch + 1 >= min_epochs
            and patience_counter >= early_stopping_patience
        ):
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if amnesiac is not None and best_amnesiac_state is not None:
            amnesiac.rollback(best_amnesiac_state)

    model.training_summary = {
        'best_epoch': int(best_epoch),
        'epochs_ran': int(epochs_ran),
        'best_val_acc': float(best_val_acc),
        'scheduler': scheduler_name,
        'early_stopping_patience': int(early_stopping_patience),
        'min_epochs': int(min_epochs),
    }

    return model, amnesiac
