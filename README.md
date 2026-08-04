# Learning with Unlearning (LwU)

Continual learning and machine unlearning in one fixed-capacity network, with
no replay buffer and no retraining from scratch.

The idea is a partition. Two importance scores are computed over the weights,
one on the retain set and one on the forget set, and the overlap of the two
masks splits the network into four zones that are then treated differently:
Zone A is frozen, Zone B is reinitialised, Zone C is updated along the
direction orthogonal to the retain gradient, and Zone D is free capacity for
the next task.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
pytest -q
```

Or with Docker, which pins CUDA, PyTorch and timm:

```bash
docker build -t lwu .
docker run --gpus all -it -v /path/to/data:/data lwu
```

The image contains the environment, not the datasets or any results.

## Running something

```bash
# Continual learning
python scripts/train_continual.py --config configs/cifar100.yaml --method lwu --scenario task

# Unlearning
python scripts/train_unlearning.py --dataset cifar100 --method lwu --forget_class 42
```

Flags override the config file, so `--lr 0.01` on the command line wins over
whatever the YAML says. Every run writes its resolved settings to
`config.json` next to its results.

Full experiments:

```bash
bash scripts/reproduce_table1.sh   # ImageNet-1K, ResNet-18, 10/20/50 tasks
bash scripts/reproduce_table2.sh   # ViT-B/16 + LoRA
```

These need the datasets mounted and take GPU time.

## Where the method lives

Everything in Section 2 of the paper is in `lwu/models/lwu.py`:

| | |
|---|---|
| SSV score, Eq. 6–7 | `compute_ssv` |
| Masks and the four zones, Eq. 8–12 | `identify_zones` |
| Zone B reinitialisation, Eq. 18 | `apply_zone_operations` |
| Zone C orthogonal projection, Eq. 19–23 | `_orthogonal_conflict_resolution` |
| Zone D masked update, Eq. 24–25 | `masked_backward`, `get_trainable_mask` |
| Test-time update, Eq. 13–17 | `test_time_update` |

`tests/test_method.py` checks these against the equations directly — that the
projected update is orthogonal to the retain gradient, that the four masks are
disjoint and cover the network, that displacement stays inside the budget, and
that test-time adaptation leaves the stored weights untouched.

## Validating the closed form

`validation/ssv_exact_validation.py` computes reference Shapley values
directly from the definition, by permutation sampling, on a network small
enough (1,024 parameters) that the full Hessian fits in memory. It then scores
the closed forms against them:

```bash
python validation/ssv_exact_validation.py --permutations 800 --seeds 0 1 2
```

About a minute per seed on CPU. Results land in
`validation/table9_results.json`.

## A note on the RTI data

`RTIDataset` in `lwu/datasets/continual_datasets.py` is a synthetic fixture. It
generates haptic-like signals with numpy so the loading path can be exercised
in CI without distributing participant data, and it is not the data behind any
reported RTI number. The real collection — 20 participants, Novint Falcon,
CHAI3D, under IRB approval — is described in Appendix H and will be released
with its data card.

## Layout

```
lwu/models/lwu.py         the method
lwu/models/vit_lora.py    ViT-B/16 with LoRA adapters
lwu/datasets/             CIFAR-100, Tiny-ImageNet, ImageNet-1K/-R/-A, RTI
lwu/baselines/            EWC, SI, LwF, SSD, Bad Teacher, UNSIR, Amnesiac
lwu/evaluation/metrics.py ACC, BWT, FWT, PS, MIA, KL
configs/                  per-dataset settings
```

`CHANGES.md` lists what changed since the version submitted for review.
