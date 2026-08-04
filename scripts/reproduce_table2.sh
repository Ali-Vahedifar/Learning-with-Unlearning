#!/usr/bin/env bash
# Reproduce Table 2: LwU + LoRA on ViT-B/16 (ImageNet-21K pretrained), CIL.
set -euo pipefail

OUT=${OUT:-results/table2}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
DATA=${DATA:-./data}
mkdir -p "$OUT"

run () {   # $1 dataset  $2 num_tasks  $3 lr
  for SEED in $SEEDS; do
    echo "[table2] dataset=$1 tasks=$2 seed=$SEED"
    python scripts/train_continual.py \
      --dataset "$1" \
      --method lwu \
      --num_tasks "$2" \
      --scenario class \
      --seed "$SEED" \
      --lr "$3" \
      --batch_size 128 \
      --lora_rank 8 --lora_alpha 16.0 \
      --tau_r 0.54 --tau_f 0.52 \
      --momentum_decay 0.95 --adaptation_rate 0.005 \
      --surprise_threshold 0.05 --stabilization_decay 0.1 \
      --eta_c 0.005 --n_c 10 --delta_c 0.5 \
      --data_dir "$DATA" \
      --output_dir "$OUT/${1}_${2}t_seed${SEED}"
  done
}

run cifar100-vit 10 1e-4
run imagenet-r   10 5e-5
run imagenet-r   20 5e-5
run imagenet-a   10 5e-5
run imagenet-a   20 5e-5

python scripts/aggregate_results.py --input "$OUT" --output "$OUT/table2_summary.csv"
