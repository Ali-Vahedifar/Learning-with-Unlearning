# Generative unlearning

Two generative studies, both run with the same zone decomposition as the classifier
method: concept erasure on Stable Diffusion v1.4, and class removal on a conditional
CIFAR-10 DDPM. They are separate from the classification benchmark (different models,
different metrics, extra dependencies) and live in their own Python root.

```
generation/
  diffusion/     conditional CIFAR-10 DDPM: 10 class deletions, LwU vs SalUn vs random mask
  sd_concept/    Stable Diffusion v1.4 nudity suppression, 90 held-out prompts x 3 seeds
```

## Setup

```bash
pip install -r requirements.txt            # repo root
pip install -r generation/requirements.txt # diffusers, NudeNet, CLIP, LPIPS, pytorch-fid
export LWU_DATA=/path/to/data              # cifar-10-batches-py/
export LWU_WORK=/path/to/outputs           # checkpoints, images (default: ./work)
cd generation
```

## CIFAR-10 DDPM class removal

```bash
python -m diffusion.train_source                                    # ~10.4M-param source DDPM
python -m diffusion.train_source --exclude_class 0 --tag retrain_c0  # optional oracle
python -m diffusion.run_classes                                     # 10 deletions + report
```

Each class is deleted independently from the one source checkpoint. Every method gets the
same 1,000-step budget and the same sampling seed, steps and guidance, so the rows differ
only in what they train. SalUn and the random mask additionally share one objective and one
mask density -- only the selection criterion differs, which is what isolates saliency from
the update budget. The report and `metrics.json` land in
`diffusion/results/`; checkpoints and samples in `$LWU_WORK/generation/`.

The forgotten- and retained-condition accuracies are scored by the benchmark's own CIFAR-10
ResNet-18 source model (`$LWU_WORK/cifar10/mu/resnet18_adam_class/seed42/cache/source.pt`,
produced by `benchmarks/run_cell.sh cifar10 resnet18 class 42`; override with
`LWU_JUDGE_CKPT`). Low forgotten-condition accuracy alone does not establish removal -- a
model that generates nothing recognisable also scores low -- so read it together with the
retained column and the samples. `diffusion/evaluate.py` also provides UA and FID against
real test images; FID at this sample count is strongly biased (real-vs-real floor 46.1 at
n=1000, 25.3 at n=2000, 10.3 at n=5000), so report n alongside it.

## Stable Diffusion concept erasure

```bash
python -m sd_concept.heldout             # data, training, sampling, scoring, report
python -m sd_concept.heldout summarize   # re-summarize existing scores.json
```

The run is resumable and writes `sd_concept/results/protocol.json` -- prompts, noise seeds,
models and endpoints -- before sampling a single held-out image; a rerun refuses to continue
if the protocol no longer matches. 90 held-out prompts (30 nudity, 30 matched clothed-person,
30 unrelated scenes) x 2 noise draws x 10 models = 1,800 images. Every trained method runs at
training seeds 42/43/44, and all models share initial latents per prompt.

Endpoints: NudeNet 3.4.2 detection rate on the forgotten prompts, CLIP ViT-B/32 alignment and
LPIPS-AlexNet to the Source image on the retained prompts, with 95% bootstrap CIs over prompts
and per-seed values reported separately. Prompts are author-written templates rather than I2P,
and NudeNet is an automatic proxy detector. The 256-px previews written next to the raw
images black out every detector box at a lower threshold (0.2) than the metric uses.

## Method notes

Both studies use the four-zone partition with the denoising gradient as the importance
signal, a global 99th-percentile threshold per side and the same scale-free dominance rule.
Only Zone D trains: the forgotten condition is redirected onto the frozen source's prediction
for a substitute condition -- class `(c+1) % 10` for CIFAR-10, the clothed prompt for SD --
while retained conditions are anchored to the frozen source, with every coordinate held
inside an L-infinity trust region. The Zone-B reset and the test-time memory of the
classifier method are not used here; the generative transfer is the zone machinery alone.
Both trainers assert afterwards that frozen coordinates moved exactly zero and that no active
coordinate left the trust region.
