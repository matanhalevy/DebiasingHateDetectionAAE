#!/bin/bash

if [[ -n "$1" ]]; then
    current_seed=$1
else
    current_seed=0
fi


if [[ -n "$2" ]]; then
    max_seeds=$2
else
    max_seeds=10
fi

echo $current_seed
echo $max_seeds

while(( $current_seed < $max_seeds ))
do
    python drive/My\ Drive/HateSpeech/benchmarking/contextual-hsd-expl/run_model.py --do_train --do_lower_case --data_dir drive/My\ Drive/HateSpeech/benchmarking/data/white_supremacy --bert_model bert-base-uncased --max_seq_length 128 --train_batch_size 32 --learning_rate 2e-5 --num_train_epochs 20 --early_stop 5 --output_dir drive/My\ Drive/HateSpeech/benchmarking/contextual-hsd-expl/runs/ws_es_reg_nb0_h1_bal_new_seed_${current_seed} --seed ${current_seed} --task_name ws --reg_explanations --nb_range 0 --sample_n 1 --hiex_add_itself --negative_weight 0.125 --reg_strength 0.1 --neutral_words_file drive/My\ Drive/HateSpeech/benchmarking/data/identity_ws_new.csv --lm_dir drive/My\ Drive/HateSpeech/benchmarking/contextual-hsd-expl/runs/lm_ws
    let current_seed++
done
