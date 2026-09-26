# Learning with Unlearning (LwU)

Code for **Learning with Unlearning**, and for every machine-unlearning and
continual-learning baseline it is compared against, each family under one protocol.
No checkpoints or results are included; every selected hyperparameter is.

LwU treats unlearning as a decomposition of the parameter space rather than a fine-tuning
objective. Retain and forget importance are estimated with diagonal Fisher information and
compared under a scale-free dominance rule, which partitions every weight into four zones:

- **Zone A (safe retain)** — importance dominated by the retained data; frozen, then repaired.
- **Zone B (pure forget)** — importance dominated by the forget set; re-initialised.
- **Zone C (conflict)** — no clear dominance; updated orthogonally to the retain gradient,
  inside an explicit displacement budget.
- **Zone D (plastic)** — unused capacity; carries an exemplar-free context memory.

After zoning, Zones A and B are repaired jointly by KL + CE self-distillation from the
frozen source model, and a causal fast-weight memory adapts the classifier at inference:
each batch is predicted before it is written, the memory lives only for the duration of an
evaluation stream, and no image, feature vector or label is stored. The method is
`lwu/LwU/lwu.py`; the stages it is built from are listed in `lwu/LwU/__init__.py`.

## Layout

```
lwu/                             machine unlearning (Python root; scripts put it on sys.path)
  LwU/                           the method
  Machine_Unlearning_baselines/  one folder per baseline (see its README)
  core/                          datasets, backbones, metrics, training helpers
  scripts/                       benchmark_unlearning.py, ulira.py, train_unlearning.py
  tests/
benchmarks/                      run_cell.sh, run_all.sh, PROTOCOL.md, HYPERPARAMETERS.md,
                                 selected_hyperparameters.csv
salun_protocol/                  SalUn-protocol study (official SalUn release + driver)
continual/                       continual learning: GTEP protocol, CL baselines, LwU adapter
generation/                      CIFAR-10 DDPM class removal, Stable Diffusion concept erasure
tools/                           aggregate_results.py (CSV over finished cells)
```

## Experimental setup → code

| Item | Where |
|---|---|
| Class-, subclass- and instance-level deletion | `lwu/scripts/benchmark_unlearning.py` (`--forget_mode`), protocol in `benchmarks/PROTOCOL.md` |
| Datasets: CIFAR-10, CIFAR-20 (CIFAR-100 superclasses), CIFAR-100, TinyImageNet-200 | `lwu/core/datasets/` (`--dataset cifar10\|cifar20\|cifar100\|tinyimagenet`) |
| Backbones: CNN (FPECNN) and ResNet-18 | `lwu/core/models/backbones.py` |
| Data splits (disjoint train/val/test and MIA calibration/selection/final subsets) | `lwu/scripts/train_unlearning.py` |
| *Original* (joint training) and *Retrain* (same recipe on D_r only) | `Benchmark._source` / `Benchmark._reference` |
| **LwU** | `lwu/LwU/` |
| Fine-tuning, Bad Teacher, Amnesiac (class deletion only), UNSIR, SSD, UniCLUN, SCRUB, SalUn, and RL, GA, ℓ1-sparse, IU, BS, BE | `lwu/Machine_Unlearning_baselines/<Method>/` |
| Searched grids, fixed settings, selected values | `benchmarks/HYPERPARAMETERS.md`, `benchmarks/selected_hyperparameters.csv` |
| Validation-only selection under a utility constraint vs Retrain; one official-test evaluation | `Benchmark.select` (`--selection utility --utility_tolerance 2.0`) |
| \|ΔD_f\|, retain/test accuracy, MIA, predictive and parameter KL, time | `lwu/core/evaluation/metrics.py` |
| Post-unlearning relearning accuracy | `Benchmark.relearn_probe` |
| U-LiRA per-example membership attack (16 shadow models) | `lwu/scripts/ulira.py` |
| SalUn protocol: RL, GA, IU, ℓ1-sparse, BS, BE, FT, SalUn, SalUn-soft (CIFAR-10, 10%/50% random forgetting, 10 trials) | `salun_protocol/` (see its README) |
| Continual learning under GTEP: SGD, Joint, EWC, SI, LwF, WSN, PEC, SpaceNet, NISPA, UniCLUN, SNV-A, LwU; CIFAR-20 / CIFAR-100 / TinyImageNet | `continual/` (see its README) |
| Generative class removal (CIFAR-10 DDPM) and concept erasure (SD v1.4) | `generation/` (see its README) |

## Setup

```bash
pip install -r requirements.txt
export LWU_DATA=/path/to/data      # cifar-10-batches-py/, cifar-100-python/, tiny-imagenet-200/
export LWU_WORK=/path/to/outputs   # checkpoints, shadow models (default: ./work)
python -m pytest -q lwu/tests
```

## Running

```bash
bash benchmarks/run_cell.sh cifar100 resnet18 class 42          # one cell-seed
bash benchmarks/run_cell.sh cifar100 resnet18 class 42 ulira    # its U-LiRA attack
PARALLEL=3 bash benchmarks/run_all.sh                           # the whole campaign
```

Each cell writes `final.json` (one test evaluation per method, with the selected
configuration and whether it met the utility budget) and `search.json` (every validation
candidate) under `$LWU_WORK/<dataset>/mu/<backbone>_adam_<mode>/seed<k>/`. The Original
model and the Retrain oracle are cached there and shared by every method in the cell, so a
cell can be interrupted and resumed without retraining anything.

A single method, dataset or mode can be run directly:

```bash
python lwu/scripts/benchmark_unlearning.py --dataset cifar20 --forget_mode subclass \
  --forget_class 0 --backbone resnet18 --methods baseline,retrain,scrub,lwu \
  --data_dir "$LWU_DATA" --output_dir "$LWU_WORK/cifar20_subclass"
```

## Tables

Once cells have finished, collect them into per-seed and 3-seed tables (no GPU):

```bash
python tools/aggregate_results.py --work "$LWU_WORK" --out results/
# results/results.csv   one row per dataset x backbone x mode x seed x method
# results/summary.csv   mean and std over seeds
```

## Notes

- `run_all.sh` defaults to three cells sharing one GPU, so `unlearn_time` includes
  contention; use `PARALLEL=1` for timings that go in a table.
- Amnesiac needs the per-batch update ledger recorded during training, so it runs for
  class deletion only and has no U-LiRA row.
- Selection is validation-only and the test set is touched once per method. Reusing a
  cell's artifacts with a changed protocol is refused rather than silently mixed: the
  `protocol` block is the cache key.
