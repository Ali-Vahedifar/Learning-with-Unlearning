# Machine unlearning baselines

One folder per method. Each exposes a class whose `unlearn(forget_loader, retain_loader)`
returns the unlearned model; `common.py` holds the shared helpers (distillation KL,
frozen copies, diagonal Fisher, paired batches, retain training). The proposed method,
LwU, is in `../LwU/`.

| Folder | Harness name | Method | Reference |
|---|---|---|---|
| `Baseline` | `baseline` | Original model, no unlearning | — |
| `Retrain` | `retrain` | retrain from scratch on D_r (gold standard) | — |
| `FineTune` | `finetune` | fine-tuning on the retain set | Golatkar et al., CVPR 2020 |
| `RandomLabel` | `rl` | random labels on D_f, true labels on D_r | Golatkar et al., CVPR 2020 |
| `GradientAscent` | `ga` | gradient ascent on D_f (NegGrad) | Thudi et al., EuroS&P 2022 |
| `L1Sparse` | `l1sparse` | fine-tuning with a decaying L1 penalty | Jia et al., NeurIPS 2023 |
| `InfluenceUnlearning` | `iu` | one WoodFisher inverse-Hessian step | Izzo et al., AISTATS 2021 |
| `BoundaryShrink` | `bs` | relabel D_f to the nearest decision region | Chen et al., CVPR 2023 |
| `BoundaryExpand` | `be` | route D_f into a shadow class, then drop it | Chen et al., CVPR 2023 |
| `BadTeacher` | `badteacher` | competent/incompetent-teacher distillation | Chundawat et al., AAAI 2023 |
| `Amnesiac` | `amnesiac` | subtracts the updates recorded for forget batches during training (class deletion only) | Graves et al., AAAI 2021 |
| `UNSIR` | `unsir` | error-maximising noise, impair then repair | Tarun et al., TNNLS 2023 |
| `SSD` | `ssd` | selective synaptic dampening | Foster et al., AAAI 2024 |
| `UniCLUN` | `uniclun` | bounded replay, dual-teacher distillation | Chatterjee et al., 2024 |
| `SCRUB` | `scrub` | teacher–student max/min steps | Kurmanji et al., NeurIPS 2023 |
| `SalUn` | `salun` | saliency mask on forget gradients + random labels | Fan et al., ICLR 2024 |

Every baseline runs under one protocol (`../scripts/benchmark_unlearning.py`): the same
source checkpoint, the same access budget, validation-only hyperparameter selection and a
single held-out test evaluation. Search grids are in `STATIC_GRIDS` there and tabulated in
`../../benchmarks/HYPERPARAMETERS.md`.
