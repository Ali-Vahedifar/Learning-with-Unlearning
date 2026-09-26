# SalUn-protocol classification study

RL, GA, IU, ℓ1-sparse, Boundary Shrink (BS), Boundary Expanding (BE), fine-tuning, SalUn and
SalUn-soft, run through the **official SalUn release** under its own protocol: CIFAR-10,
the release's ResNet-18, random forgetting of 10% and 50% of the training set.

- `salun_official/` is the release (MIT, see its `LICENSE`), commit
  `a6e89fd17ff737c24c6aa16c9574469a234ffd91`, with one local change in `local.patch`: an
  unused import of a function that does not exist in the release.
- `classification.py` drives it. Source and retrain: 182 epochs of cosine SGD (lr 0.1,
  momentum 0.9, weight decay 5e-4, batch 256). Metrics: UA (= 100 − forget accuracy), RA,
  TA and the release's confidence-SVC MIA.
- **Selection.** Pilot seed 1 searches a coarse grid inside the published ranges (`GRID`)
  and picks, per method, the smallest mean absolute gap to the paired retrain model on
  UA/RA/validation accuracy/MIA. The choice is frozen before the test trials.
- **Test trials.** Seeds 2-11, each with its own source and paired retrain model.
- `selected/forget10.json`, `selected/forget50.json`: the recorded selections.

```bash
export LWU_DATA=/path/to/data LWU_WORK=/path/to/outputs
python preflight.py                  # every released entry point on 4 images (GPU)
python run_study.py                  # tune on seed 1, then seeds 2-11, both ratios
python run_study.py --recorded       # skip tuning, use selected/*.json
python summarize.py                  # -> $LWU_WORK/salun_protocol/summary.md
```

Deviations from the release, all deliberate: the train/validation split follows the run
seed (the release fixes seed 1 for it); retraining uses the cosine schedule of the paper's
appendix rather than the release's default milestones; validation/test disable
augmentation; BE's temporary eleventh class is removed before ten-class evaluation;
SalUn-soft is the release's `RL_proximal` with an explicit mask ratio, whose epoch-wise
anchor schedule differs from the idealised proximal update. Exact published selected
hyperparameters were not released, hence the explicit grid.
