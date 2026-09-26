#!/usr/bin/env bash
# The whole campaign: 3 datasets x 2 backbones x 3 deletion modes x 3 seeds,
# then the U-LiRA attack on seed 42.
# Usage: PARALLEL=3 DATASETS="cifar10 cifar20 cifar100 tinyimagenet" run_all.sh
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PARALLEL=${PARALLEL:-3}
DATASETS=${DATASETS:-"cifar10 cifar20 cifar100"}
for DS in $DATASETS; do
  for SEED in 42 43 44; do
    for MODE in instance class subclass; do
      for BB in cnn resnet18; do
        while [ "$(jobs -pr | wc -l)" -ge "$PARALLEL" ]; do wait -n; done
        { bash "$HERE/run_cell.sh" "$DS" "$BB" "$MODE" "$SEED" > /dev/null
          echo "[$(date -u +%FT%TZ)] $DS $BB $MODE seed$SEED exit=$?"; } &
      done
    done
  done
done
wait
for DS in $DATASETS; do
  for MODE in class subclass instance; do
    for BB in resnet18 cnn; do
      while [ "$(jobs -pr | wc -l)" -ge "$PARALLEL" ]; do wait -n; done
      { bash "$HERE/run_cell.sh" "$DS" "$BB" "$MODE" 42 ulira > /dev/null
        echo "[$(date -u +%FT%TZ)] ulira $DS $BB $MODE exit=$?"; } &
    done
  done
done
wait
echo "CAMPAIGN COMPLETE"
