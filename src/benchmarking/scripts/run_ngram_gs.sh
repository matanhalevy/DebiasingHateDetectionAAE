#!/bin/bash

datasets=(hate harassment davidson founta waseem golbeck)
epochs=(10 100 1000)
batch_sizes=(16 32 64)
learning_rates=(2e-3 2e-5 5e-5)
current_seed=42

for dataset in "${datasets[@]}"
do
  for epoch in "${epochs[@]}"
  do
    for batch_size in "${batch_sizes[@]}"
    do
      python drive/My\ Drive/HateSpeech/benchmarking/baselines/run_model.py --task_name ngram --dataset $dataset --is_local False --is_gridsearch True --seed $current_seed --learning_rate $learning_rate --batch_size $batch_size --epochs $batch_size --max_seq_length 128 --output_dir drive/My\ Drive/HateSpeech/benchmarking/baselines/runs/log_reg_glove_gs --do_train --do_eval
    done
  done
done
