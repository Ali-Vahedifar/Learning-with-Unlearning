#!/usr/bin/env bash
# One benchmark cell-seed.
# Usage: run_cell.sh <cifar10|cifar20|cifar100|tinyimagenet> <cnn|resnet18> <class|subclass|instance> <seed> [ulira]
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
DS=$1; BACKBONE=$2; MODE=$3; SEED=$4; STAGE=${5:-benchmark}
DATA=${LWU_DATA:-$REPO/data}; WORK=${LWU_WORK:-$REPO/work}
OUT=$WORK/$DS/mu/${BACKBONE}_adam_${MODE}/seed$SEED

case $DS in
  cifar10)  FORGET=0;  EXTRA=(--epochs 100 --early_stopping_patience 10 --min_epochs 20
                              --fpecnn_hidden_size 64
                              --protocol_version cifar10-mu-protocol-v3) ;;
  cifar20)  FORGET=0;  EXTRA=(--epochs 150 --early_stopping_patience 15 --min_epochs 30
                              --fpecnn_hidden_size 256
                              --protocol_version cifar20-mu-protocol-v3) ;;
  cifar100) FORGET=42; EXTRA=(--epochs 150 --early_stopping_patience 15 --min_epochs 30
                              --fpecnn_hidden_size 512
                              --protocol_version cifar100-mu-protocol-v3) ;;
  tinyimagenet) FORGET=0; EXTRA=(--epochs 150 --early_stopping_patience 15 --min_epochs 30
                              --fpecnn_hidden_size 512
                              --protocol_version tinyimagenet-mu-protocol-v3) ;;
  *) echo "unknown dataset $DS" >&2; exit 2 ;;
esac
# Instance deletion removes 10% of the training set (45k CIFAR / 90k TinyImageNet
# training images after the validation split); class/subclass remove one label.
NUM_FORGET=4500; [ "$DS" = tinyimagenet ] && NUM_FORGET=9000
if [ "$MODE" = instance ]; then SPLIT=(--num_forget $NUM_FORGET); else SPLIT=(--forget_class "$FORGET"); fi

ARGS=(--dataset "$DS" --forget_mode "$MODE" "${SPLIT[@]}" --backbone "$BACKBONE" --optimizer adam
      "${EXTRA[@]}" --batch_size 64 --source_lr 0.001 --retain_ratio 0.1
      --selection utility --utility_tolerance 2.0 --relearn_epochs 5
      --seed "$SEED" --device "${DEVICE:-cuda:0}" --data_dir "$DATA" --output_dir "$OUT")
mkdir -p "$OUT"

if [ "$STAGE" = ulira ]; then
  python -u "$REPO/lwu/scripts/ulira.py" --shadows 16 "${ARGS[@]}" --num_workers 0 2>&1 | tee -a "$OUT/ulira.log"
else
  METHODS=baseline,retrain,finetune,rl,ga,l1sparse,iu,bs,be,badteacher,unsir,ssd,uniclun,scrub,salun,lwu
  # Amnesiac needs the per-batch ledger recorded during source training, which
  # only exists for whole-class deletion.
  [ "$MODE" = class ] && METHODS="$METHODS,amnesiac"
  python -u "$REPO/lwu/scripts/benchmark_unlearning.py" "${ARGS[@]}" --phase all --methods "$METHODS" 2>&1 | tee -a "$OUT/run.log"
fi
