#!/bin/bash
# run with best epoch, batch size, and learning rate after running GS
max_seeds=10
current_seed=0

while (($current_seed < $max_seeds)); do
  python drive/My\ Drive/HateSpeech/benchmarking/baselines/run_model.py --task_name tf_idf --dataset harassment --seed $current_seed --learning_rate 2e-5 --batch_size 32 --epochs 100 --max_seq_length 128 --output_dir drive/My\ Drive/HateSpeech/benchmarking/baselines/runs/log_reg_tfidf_seed_$current_seed --do_train --do_eval
  let current_seed++
done
