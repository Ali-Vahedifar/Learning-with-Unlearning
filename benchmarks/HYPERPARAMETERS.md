# Hyperparameters

Two kinds of numbers live here: what is **searched** (a Cartesian grid, selected on
validation data only, one final test evaluation for the winner) and what is **fixed** for
every method and dataset. Both are written into each run's `search.json` / `final.json`,
so a result can always be traced back to the configuration that produced it. The grids
themselves are `STATIC_GRIDS` and `Benchmark._lwu_grid` in
`lwu/scripts/benchmark_unlearning.py`.

**Selected values.** `selected_hyperparameters.csv` lists the configuration each method
selected in every reported cell and seed (CIFAR-10/-20/-100 × CNN/ResNet-18 × class/
subclass/instance × seeds 42-44), with its `feasible` flag. The per-dataset training
settings those cells ran with are the ones in `run_cell.sh`. It covers the methods run in that campaign: fine-tuning, Bad Teacher,
Amnesiac, UNSIR, SSD, UniCLUN, SCRUB, SalUn and LwU. RL, GA, ℓ1-sparse, IU, BS and BE have
grids below but were reported under the SalUn protocol instead; their selections are in
`../salun_protocol/selected/`. The continual-learning (GTEP) winners are in
`../continual/selected_hyperparameters_cifar20.csv`.

## Fixed for the whole benchmark

| Setting | Value | Flag |
|---|---|---|
| Source/Retrain optimiser | Adam, lr 1e-3, weight decay 0 | `--optimizer adam --source_lr 1e-3` |
| Scheduler | ReduceLROnPlateau on validation loss, factor 0.5, patience 5 | `--source_scheduler plateau` |
| Batch size | 64 | `--batch_size 64` |
| Epoch cap / early-stopping patience / minimum epochs | 100/10/20 (CIFAR-10), 150/15/30 (CIFAR-20, CIFAR-100) | `--epochs --early_stopping_patience --min_epochs` |
| Retain access for post-hoc methods | 10% of D_r | `--retain_ratio 0.1` |
| Selection rule | utility-constrained, tolerance 2.0 points vs Retrain | `--selection utility --utility_tolerance 2.0` |
| Relearn probe | 5 Adam epochs, lr 1e-4, on D_f | `--relearn_epochs 5 --relearn_lr 1e-4` |
| Instance deletion size | 4,500 images (10%) | `--num_forget 4500` |
| U-LiRA shadow models | 16 | `--shadows 16` |
| Seeds | 42, 43, 44 | `--seed` |

## Searched grids, baselines

| Method | Grid | Candidates |
|---|---|---:|
| `finetune` | lr {1e-4, 1e-3, 1e-2} × epochs {1, 3, 5, 10} | 12 |
| `rl` | lr {1e-4, 1e-3, 1e-2} × epochs {3, 5, 10} | 9 |
| `ga` | lr {1e-5, 1e-4, 1e-3} × epochs {1, 3, 5} | 9 |
| `l1sparse` | lr {1e-4, 1e-3, 1e-2} × epochs {5, 10} × alpha {1e-5, 5e-5, 1e-4} | 18 |
| `iu` | alpha {0.1, 0.5, 1, 5, 10, 20} | 6 |
| `bs` | lr {1e-5, 1e-4, 1e-3} × epochs {1, 3, 5} × bound {0.1} | 9 |
| `be` | lr {1e-5, 1e-4, 1e-3} × epochs {1, 3, 5} | 9 |
| `badteacher` | temperature {1, 2, 4} × epochs {1, 3, 5, 10} × lr {1e-4, 1e-3, 1e-2} | 36 |
| `unsir` | impair_epochs {1, 2} × repair_epochs {1, 2, 3} × impair_lr {1e-3, 1e-2, 1e-1} × repair_lr {1e-3, 1e-2} | 36 |
| `ssd` | alpha {1, 5, 10, 50, 100} × dampening_constant {0.1, 0.5, 1, 5, 10} | 25 |
| `uniclun` | lr {1e-4, 1e-3, 1e-2} × epochs {3, 5, 10} × buffer_size {250, 500, 1000} | 27 |
| `scrub` | msteps {1, 2, 3} × lr {1e-4, 5e-4, 1e-3, 5e-3} × epochs {5, 10} × max_grad_norm {5.0} | 24 |
| `salun` | sparsity {0.1, 0.3, 0.5, 0.7, 0.9} × lr {1e-4, 1e-3, 1e-2} × epochs {3, 5, 10} | 45 |

`baseline`, `retrain` and `amnesiac` have no grid: the first two are defined by the
protocol, and Amnesiac replays the recorded training updates for the forget batches.

Fixed inside the baselines: SGD momentum 0.9 and weight decay 5e-4 for RL, GA, l1-sparse,
BS and BE (the optimiser of the official releases); RL's retain weight alpha = 1.0;
l1-sparse's L1 coefficient decays linearly to 0 over the epoch budget; BS's FGSM bound
0.1 in normalised input space; IU's WoodFisher damping N = 1000 over at most 1,000 single
retain examples.

## Searched grid, LwU

| Knob | Values | What it controls |
|---|---|---|
| `forget_quantile` | {0.99} for class deletion, {0.9, 0.99} for subclass/instance | how much of the forget-importance mass counts as forget-active |
| `dominance_margin` | {1.0, 3.0} | how much one importance map must exceed the other to own a weight; 1.0 makes dominance exhaustive and empties Zone C |
| `distill_epochs` | {10, 30} | repair budget after zeroing Zone B |
| `distill_lr` | {1e-3, 1e-2} | repair learning rate |
| `weight_align` | {False, True} | re-align repaired weights to the pre-unlearning norm |
| `memory_preset` | {`m01_s025_c08`, `m10_s100_c08`} | test-time memory strength (table below) |
| `sdft_lambda` | {0.5} | scale of the extra Zone-C self-distillation pass |
| `instance_forget_priority` | {False} | kept off: reinitialising every forget-active weight pinned accuracy at chance even with the full retain set and 30 repair epochs |

32 candidates for class deletion, 64 for subclass and instance.

### LwU settings that are fixed, not searched

| Setting | Value |
|---|---|
| importance estimator | diagonal Fisher, per-instance on the forget side for instance deletion |
| zoning rule | dominance, scale-free (each map normalised by its own threshold) |
| Zone C rule | orthogonal update, step `eta_c` 1e-2, `n_c` 5 steps, displacement budget `delta_c` 0.1 |
| `retain_quantile` | 0.9 |
| distillation temperature / CE weight | 4.0 / 1.0 |
| context window / lr / temperature / decay / momentum / steps | 1 / 0.0 / 0.25 / 1.0 / 0.9 / 1 |
| logit masking | on for class deletion only |
| memory persistence across streams | off (the memory is reset for every evaluation stream) |
| memory decay / momentum / max norm | 0.995 / 0.9 / 5.0 |

### Memory presets

| Preset | `memory_lr` | `memory_scale` | `memory_confidence` |
|---|---:|---:|---:|
| `m01_s025_c00` | 0.01 | 0.25 | 0.0 |
| `m01_s025_c08` | 0.01 | 0.25 | 0.8 |
| `m10_s100_c08` | 0.10 | 1.00 | 0.8 |

`m01_s025_c00` (no confidence gate) is defined for the ablation; the searched grid uses
the two gated presets.
