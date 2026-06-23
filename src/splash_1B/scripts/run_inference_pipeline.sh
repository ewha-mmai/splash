#!/bin/bash

SPARSITIES=(60)
CHECKPOINTS=(2428)

for S in "${SPARSITIES[@]}"; do
  for C in "${CHECKPOINTS[@]}"; do

    bash src/splash_1B/scripts/run_intern_inference.sh $S $C
    bash src/splash_1B/scripts/run_intern_evaluation.sh $S $C

  done
done
