#!/bin/bash

echo "running davidson glove"
bash HateSpeech/benchmarking/baselines/scripts/run_eval_lr.sh glove davidson 1000 32 0.002
echo "running founta glove"
bash HateSpeech/benchmarking/baselines/scripts/run_eval_lr.sh glove founta 100 16 5e-5
echo "running golbeck glove"
bash HateSpeech/benchmarking/baselines/scripts/run_eval_lr.sh glove golbeck 10 32 5e-5
echo "running harassment glove"
bash HateSpeech/benchmarking/baselines/scripts/run_eval_lr.sh glove harassment 100 16 5e-5
echo "running hate glove"
bash HateSpeech/benchmarking/baselines/scripts/run_eval_lr.sh glove hate 100 64 2e-5
echo "running davidson ngram"
bash HateSpeech/benchmarking/baselines/scripts/run_eval_lr.sh ngram davidson 100 16 0.002
echo "running founta ngram"
bash HateSpeech/benchmarking/baselines/scripts/run_eval_lr.sh ngram founta 10 64 0.002
echo "running golbeck ngram"
bash HateSpeech/benchmarking/baselines/scripts/run_eval_lr.sh ngram golbeck 1000 32 5e-5
echo "running harassment ngram"
bash HateSpeech/benchmarking/baselines/scripts/run_eval_lr.sh ngram harassment 10 64 0.002
echo "running hate ngram"
bash HateSpeech/benchmarking/baselines/scripts/run_eval_lr.sh ngram hate 1000 16 5e-5
echo "running davidson tf_idf"
bash HateSpeech/benchmarking/baselines/scripts/run_eval_lr.sh tf_idf davidson 1000 16 0.002
echo "running founta tf_idf"
bash HateSpeech/benchmarking/baselines/scripts/run_eval_lr.sh tf_idf founta 1000 16 0.002
echo "running golbeck tf_idf"
bash HateSpeech/benchmarking/baselines/scripts/run_eval_lr.sh tf_idf golbeck 1000 32 0.002
echo "running harassment tf_idf"
bash HateSpeech/benchmarking/baselines/scripts/run_eval_lr.sh tf_idf harassment 10 16 0.002
echo "running hate tf_idf"
bash HateSpeech/benchmarking/baselines/scripts/run_eval_lr.sh tf_idf hate 1000 64 0.002
