#!/bin/bash

task=$1
dataset=$2
epoch=$3
batch_size=$4
learning_rate=$5

if [[ -n "$6" ]]; then
    current_seed=$5
else
    current_seed=0
fi


if [[ -n "$7" ]]; then
    max_seeds=$6
else
    max_seeds=10
fi

echo "task is ${task}"
echo "dataset is ${dataset}"
echo "epoch is ${epoch}"
echo "batch_size is ${batch_size}"
echo "learning_rate is ${learning_rate}"

while(( $current_seed < $max_seeds ))
do
    python /nethome/mhalevy3/HateSpeech/benchmarking/baselines/run_model.py --task_name $task --dataset $dataset --seed $current_seed --learning_rate $learning_rate --batch_size $batch_size --epochs $epoch --max_seq_length 128 --output_dir /nethome/mhalevy3/HateSpeech/benchmarking/baselines/runs/log_reg_glove_gs --do_train --do_eval
    python /nethome/mhalevy3/HateSpeech/benchmarking/baselines/run_model.py --task_name $task --dataset $dataset --seed $current_seed --learning_rate $learning_rate --batch_size $batch_size --epochs $epoch --max_seq_length 128 --output_dir /nethome/mhalevy3/HateSpeech/benchmarking/baselines/runs/log_reg_glove_gs --test
    let current_seed++
done
