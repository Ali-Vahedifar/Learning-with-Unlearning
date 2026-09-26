# Classification benchmark protocol

Implemented by `lwu/scripts/benchmark_unlearning.py`; one cell-seed is launched by
`run_cell.sh`, the whole grid by `run_all.sh`. Every setting below is written into the
`protocol` block of each `final.json`, and the per-method search grids are listed in
`HYPERPARAMETERS.md`.

## Shared across datasets

- **Source ("Original") training.** Adam, lr 1e-3 (betas 0.9/0.999, weight decay 0),
  batch 64, ReduceLROnPlateau on validation loss (factor 0.5, patience 5), early
  stopping with the best-validation checkpoint restored. Train augmentation
  RandomCrop(32, pad 4) + horizontal flip; validation/test normalise only.
- **Splits.** The 50,000 training images are split 90/10 stratified into 45,000 train /
  5,000 validation. The 10,000-image official test set is evaluated **once per method,
  after selection**.
- **Retrain** uses the identical recipe on D_r only (same optimiser, scheduler,
  validation and early stopping). Every method in a cell starts from one byte-identical
  source checkpoint (SHA-256 recorded per result).
- **Access.** Post-hoc methods see D_f plus 10% of D_r (`--retain_ratio 0.1`);
  fine-tuning and l1-sparse see all of D_r. Amnesiac additionally needs the per-batch
  update ledger recorded during source training, so it runs for class deletion only.
  Each method's access is recorded next to its numbers (`access` in `final.json`).
- **Selection (`--selection utility`).** Among grid candidates whose retain **and**
  test accuracy on validation data stay within 2.0 points of Retrain, pick the one whose
  forget accuracy is closest to Retrain's (ties: higher retain accuracy). If nothing is
  feasible, the least damaging candidate is reported and flagged `feasible: false`.
  The reported |ΔD_f| is therefore not the tuned quantity.
- **Gradient clipping.** SCRUB's max-step uses gradient-norm clipping 5.0; without it it
  diverged at every lr ≥ 5e-4 on the CNN backbone.
- **Relearn probe.** After the test evaluation, 5 Adam epochs at lr 1e-4 on D_f; forget
  accuracy after each epoch is stored as `relearn_forget_acc`. A method that only
  suppressed its forget outputs snaps back within one epoch.
- **Seeds** 42, 43, 44. **U-LiRA** (`lwu/scripts/ulira.py`): seed 42, 16 shadow models
  per cell, per-example Gaussian likelihood ratio with shared variance; positives are the
  method's final model, negatives the Retrain model on the same D_f examples. Amnesiac
  has no per-shadow equivalent (its ledger is recorded during source training) and is
  excluded, as is Retrain, which is the attack's reference.

## Deletion modes (`--forget_mode`)

| Mode | What is removed | How the forget set is selected |
|---|---|---|
| `class` | one whole label | `--forget_class` against the training label |
| `subclass` | one fine class inside a superclass, while its siblings stay | model trains/evaluates on the coarse label; `.targets` keeps the fine label |
| `instance` | `--num_forget` individual images (4,500 = 10%), class-balanced | random, stratified, disjoint from the MIA calibration subsets |

## Per-dataset settings

| | CIFAR-10 | CIFAR-20 | CIFAR-100 | TinyImageNet |
|---|---|---|---|---|
| label space | 10 classes | 20 CIFAR-100 superclasses | 100 classes | 200 classes, 64×64 |
| epoch cap / patience / min | 100 / 10 / 20 | 150 / 15 / 30 | 150 / 15 / 30 | 150 / 15 / 30 |
| CNN (FPECNN) head width | 64 | 256 | 512 | 512 |
| class deletion | class 0 | superclass 0 | class 42 | class 0 |
| subclass deletion | fine class 0 inside the 5×2 grouping of `core/datasets/cifar10_coarse.py` | fine class 0 inside its superclass | fine class 42 inside its superclass; model trains on the 20 superclasses | fine class 0 inside its WordNet group; model trains on the 56 groups of `core/datasets/tinyimagenet.py` |
| instance deletion | 4,500 images (10%), class-balanced | same | same | 9,000 images (10%) |

TinyImageNet: the 100,000 training images are split 90/10 into train/validation, the
official 10,000-image validation set is the test set (its test split is unlabelled), and
augmentation is RandomCrop(64, pad 8) + flip with ImageNet normalisation. The loader sorts
the raw archive's flat `val/images` into class folders on first use. No TinyImageNet cell
has been run yet; its settings follow the CIFAR-100 rule.

Backbones: FPECNN ("CNN", ~2M parameters) and ResNet-18 (small-input stem, ~11M).

## Metrics (`lwu/core/evaluation/metrics.py`, `UnlearningEvaluator`)

| key in `final.json` | paper column |
|---|---|
| `forget_acc` | D_f accuracy; tables report \|ΔD_f\| = \|D_f(method) − D_f(Retrain)\| per seed |
| `retain_acc`, `test_acc` | D_r and test accuracy |
| `mia`, `mia_auc` | membership-inference attack (threshold and score direction learned on independent calibration subsets) |
| `output_kl_divergence` | predictive KL to Retrain |
| `kl_divergence` | parameter-distribution KL to Retrain |
| `unlearn_time` | wall-clock seconds, method preparation included (e.g. SSD's Fisher); Retrain = its training time |
| `relearn_forget_acc` | D_f accuracy after each relearn epoch |
| `ulira.json` | U-LiRA AUC, TPR@1%FPR, TPR@0.1%FPR, balanced accuracy |

LwU is evaluated with its test-time memory active (`test_time_adapt`), which is an
inference-time fast weight: each batch is predicted before it is written, the memory
lives only for the evaluation stream, and the stored parameters never change.

Timing caveat: `run_all.sh` defaults to `PARALLEL=3`, so unlearning times include
contention from unrelated cells. Use `PARALLEL=1` for publication timings.
