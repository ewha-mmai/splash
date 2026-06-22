#!/bin/bash

SPARSITIES=(60)
CHECKPOINTS=(1214)

for S in "${SPARSITIES[@]}"; do
  for C in "${CHECKPOINTS[@]}"; do

    bash src/splash_3B/scripts/run_mask_inference.sh $S $C
    bash src/splash_3B/scripts/run_mask_evaluation.sh $S $C

  done
done