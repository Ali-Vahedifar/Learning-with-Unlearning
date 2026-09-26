# Continual learning: the GTEP campaign

LwU as a continual learner, and every continual-learning baseline it is compared against,
under one protocol (GTEP: disjoint class halves, D_HT for tuning and D_E for reporting,
R = 30 sampled configurations × 3 seeds, harmonic-mean selection, exclusive-GPU cost runs).
Full protocol in [docs/PROTOCOL.md](docs/PROTOCOL.md), metric definitions in
[docs/METRICS.md](docs/METRICS.md), cost measurements in [docs/COSTS.md](docs/COSTS.md),
per-method notes in [docs/BASELINES.md](docs/BASELINES.md), reference repositories in
[THIRD_PARTY.md](THIRD_PARTY.md).

| Group | Methods | Folder |
|---|---|---|
| Bounds | SGD (lower), Joint training (upper) | `SGD/`, `Joint/` + `joint_audited.py` |
| Regularisation | EWC, SI, LwF | `EWC/`, `SI/`, `LwF/` |
| Sparse / architecture | WSN (Task-IL only), PEC (Class-IL only), SpaceNet, NISPA | `WSN/`, `PEC/`, `SpaceNet/`, `NISPA/` |
| CL + unlearning | UniCLUN | `UniCLUN/` |
| Shapley valuation | SNV-A | `SNV/snv_adaptive.py` on `SNV/snv_core.py` |
| **LwU** | zone stage of `../lwu/LwU/` in CL-only mode | `LwU/lwu.py` |

Under GTEP there is no forget set, so LwU runs as its zone decomposition: Zone A of each
finished task is frozen and the test-time update runs at evaluation. The later LwU stages
(zone-B reset, repair, self-distillation, causal memory) act on a forget set and are
exercised by the unlearning benchmark, not here.

## Layout

```
audited_gtep.py      protocol: splits, search spaces (SPACE), one run, cost-ledger hooks
snv_adaptive_run.py  runs SNV-A through the same worker
campaign/
  run_campaign.py    tuning (30 configs x 3 seeds), then 3 clean D_E runs per winner
  build_report.py    metrics / cost / hyperparameter tables (CSV, XLSX, Markdown, HTML)
metrics.py           ACC, BWT, FWT, PS (+ P, S, AF)
audit_cost.py        cost ledger: GPU-hours, peak memory, parameters, GFLOPs, latency, energy
datasets.py          CIFAR-100, CIFAR-20 (superclasses), TinyImageNet-200
models.py            backbones (CIFAR ResNet-18 and variants)
cl_base.py, baselines.py, train.py, training_policy.py, utils.py, cost.py, inrun.py
<Method>/            one folder per method
selected_hyperparameters_cifar20.csv   the winners of the reported campaign
```

## Run

```bash
pip install -r requirements.txt
export GTEP_DATA_ROOT=$LWU_DATA                   # CIFAR-100 downloads there on first use
bash scripts/smoke_test.sh                        # every method, 2 tasks, 1 epoch (CPU)
GTEP_DEVICE=cpu GTEP_GPU_UUIDS=none python -m pytest -q tests

# The reported campaign: CIFAR-20, 5 tasks x 2 superclasses per half, one GPU
GTEP_DATASET=cifar20 GTEP_NUM_WORKERS=0 python campaign/run_campaign.py \
    --out runs/cifar20 --gpus 0 --pack 16
python campaign/build_report.py --campaign runs/cifar20 --out reports/cifar20

# One run (method, scenario, half, seed) with a recorded winner
GTEP_PROTOCOL=legacy GTEP_DATASET=cifar20 python audited_gtep.py --one --method lwu \
    --scenario class_il --half 2 --seed 42 --epochs 200 --patience 15 --tasks 5 --out runs/lwu \
    --config '{"lr":0.0007900751110328372,"tau_r":0.0003,"momentum_decay":0.9,"adaptation_rate":0.001}'
```

`GTEP_DATASET` also accepts `cifar100` (10 tasks × 5 classes per half) and `tinyimagenet`.
`selected_hyperparameters_cifar20.csv` holds every winner with its D_HT score and D_E
accuracy; SNV-A's search had not finished when it was exported.

Seeding note: a run builds the loaders of all `--tasks` tasks before the model, so the
same seed with a different `--tasks` gives a different initialisation. Reproduce a
campaign run with its own `--tasks`.
