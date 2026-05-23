#!/bin/bash
CORRUPTION=$1
CORRUPT_POR=$2
TRAIN_SIZE=$3
VAL_SIZE=2000   # as in paper
SEED=42
CUDA=1

echo "Running: $CORRUPTION with $CORRUPT_POR corruption, train=$TRAIN_SIZE, val=$VAL_SIZE"
python value_cifar10.py \
    --hierarchical \
    --random_seed=${SEED} \
    --corrupt_por=${CORRUPT_POR} \
    --corruption_type=${CORRUPTION} \
    --cache_l2l \
    --tag=sava_${CORRUPTION}_train${TRAIN_SIZE}_seed${SEED} \
    --cuda_num=${CUDA} \
    --train_dataset_sizes ${TRAIN_SIZE} \
    --val_dataset_size ${VAL_SIZE} \
    --evaluate \
    2>&1 | tee logs/sava_${CORRUPTION}_${TRAIN_SIZE}.log