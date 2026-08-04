# Changes to this artifact

This file records every change made to the code after the review period began,
so that each one can be checked against the previous version.

## Fixes to code paths that could not execute

- `scripts/train_continual.py`, `scripts/train_unlearning.py`: the calls to
  `get_dataset` passed `data_dir=` and `scenario=`, neither of which is
  accepted by the dataset constructors. The factory now accepts `data_dir` as
  an alias for `root` and records `scenario` on the dataset object.
- `lwu/datasets/continual_datasets.py`: `cifar100-vit`, used by
  `scripts/reproduce_table2.sh`, was accepted by the argument parser but was
  not present in the dataset registry. It is now registered, and
  `CIFAR100Dataset` takes an `image_size` argument (224 with ImageNet
  normalisation for the ViT setting, 32 with CIFAR normalisation otherwise).
- `scripts/train_unlearning.py`: called `dataset.split_forget_retain`, which
  does not exist; the method is `get_forget_retain_loaders`. Also corrected
  the `retrain_from_scratch` call, which passed argument names the function
  does not take.
- `scripts/train_unlearning.py`, `scripts/train_continual.py`: backbone input
  dimension for feature-vector datasets is now read from the data rather than
  from a hardcoded config entry, which disagreed with the loader.

## Configuration

- `scripts/train_continual.py`: added `--config`, so the YAML files in
  `configs/` are actually loaded. `load_config` existed but was never called.
  Precedence is explicit command-line flags > config file > argparse defaults.
  Applied values are printed and written to each run's `config.json`.

## Metrics

- `lwu/evaluation/metrics.py`: the Plasticity-Stability metric previously
  computed plasticity as the mean of the accuracy-matrix diagonal and
  stability as a mean retention ratio. That follows the reference
  implementation of the work that introduced PS, not the definition stated in
  this paper. It now computes

      P = 1/(T-1) * sum_t (A[t,t] - A[t-1,t]) / (1 - A[t-1,t])
      S = 1 - 1/(T-1) * sum_t (A[t,t] - A[T,t])
      PS = 2PS / (P + S)

  `S = 1 + BWT` holds by construction and is asserted in the test suite.

- Computing P requires A[t-1, t], the accuracy on a task before training on
  it. The evaluator never measured this, so the entry is not present in
  previously produced accuracy matrices and cannot be recovered from them.
  `ContinualLearningEvaluator.evaluate_before_task` now measures it, and both
  training loops call it. If the entries are missing, `get_ps` raises rather
  than returning a value computed from a different definition.

- `evaluate_task` accepted a `task_id` argument documented as being for the
  Task-IL scenario and did not use it: the prediction was taken as an argmax
  over the full label space regardless of scenario, which is the Class-IL
  protocol. It now accepts `task_classes` and restricts the argmax to the
  evaluated task's classes when the scenario is Task-IL. `scenario` is
  threaded from the dataset through the evaluator.

## RTI fixture

- `RTIContinualDataset` was hardcoded to 5 classes with one action per task
  and so could not instantiate the 10-task protocol. Class count now follows
  the requested protocol, with an explicit `num_classes` override for the
  single-task unlearning setting.
- `RTIDataset` remains a SYNTHETIC fixture for exercising the loading path in
  CI. It is not the data behind any reported RTI result. Its feature extractor
  emits 10 features per signal, which differs from the 25 assumed by the
  configuration; the model now takes this dimension from the data.

## Environment

- `Dockerfile` added, pinning CUDA, PyTorch and timm. The image contains the
  environment only: no datasets, no checkpoints, no results.
- `requirements.txt` changed from lower bounds to exact pins.
- `tests/test_repository.py` added: nine tests covering the dataset factory,
  the metric formulas, and Task-IL masking. `pytest -q` passes.

## Method verification

- `tests/test_method.py` added. It checks the implementation against the
  properties Section 2 claims: that SSV equals the first-order term plus the
  diagonal curvature term, that masks threshold |phi| rather than phi, that
  the four zones partition the parameter space exactly, that the Zone C update
  is orthogonal to the retain gradient and never amplifies the raw update,
  that displacement stays inside the budget, that Zone A receives no gradient
  during training, and that test-time adaptation leaves the stored weights
  unchanged. `pytest -q` runs 17 tests.

## SSV validation harness

- `validation/ssv_exact_validation.py` added. It computes reference Shapley
  values directly from Eq. 5 by permutation sampling on a 1,024-parameter
  network, where the full Hessian (1,048,576 entries, 4.2 MB fp32) can be
  formed exactly, and compares two closed forms against them by Spearman rank
  correlation: the diagonal form used in deployment, and the full form
  including the cooperative interaction sum.

- Result over three seeds, 800 permutations each (819,200 utility evaluations
  per seed), in `validation/table9_results.json`:

      diagonal only                  0.757 +/- 0.017
      with cooperative interactions  0.668 +/- 0.010

  The diagonal form is consistent with the value reported in Table 9. The
  cooperative term reduces rather than improves the correlation in this
  harness, which does not match the reported improvement to 0.895. The
  discrepancy is under investigation; possible sources include the removal
  operation used for the utility function, the choice of network and task, the
  path weights w_ij, and the use of the exact Hessian rather than the
  empirical Fisher.

- The harness runs on CPU in roughly one minute per seed:

      python validation/ssv_exact_validation.py --permutations 800 --seeds 0 1 2
