#!/usr/bin/env bash
# Reproduce Table 1: ImageNet-1K, ResNet-18, CIL + TIL at 10/20/50 tasks.
# Writes per-seed JSON so every reported cell is traceable to its run.
set -euo pipefail

OUT=${OUT:-results/table1}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
DATA=${DATA:-./data}
mkdir -p "$OUT"

for TASKS in 10 20 50; do
  for SCENARIO in class task; do
    for SEED in $SEEDS; do
      echo "[table1] tasks=$TASKS scenario=$SCENARIO seed=$SEED"
      python scripts/train_continual.py \
        --dataset imagenet1k \
        --method lwu \
        --num_tasks "$TASKS" \
        --scenario "$SCENARIO" \
        --seed "$SEED" \
        --lr 0.001 \
        --batch_size 128 \
        --tau_r 0.54 --tau_f 0.52 \
        --momentum_decay 0.95 --adaptation_rate 0.005 \
        --surprise_threshold 0.05 --stabilization_decay 0.1 \
        --eta_c 0.005 --n_c 10 --delta_c 0.5 \
        --data_dir "$DATA" \
        --output_dir "$OUT/lwu_${SCENARIO}_${TASKS}t_seed${SEED}"
    done
  done
done

python scripts/aggregate_results.py --input "$OUT" --output "$OUT/table1_summary.csv"
