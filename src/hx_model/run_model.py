import pandas as pd
import numpy as np
import os
import gc
import subprocess
from tokenizers.implementations import BertWordPieceTokenizer
from tokenizers.processors import BertProcessing
from transformers import BertConfig, BertForMaskedLM, LineByLineTextDataset, BertTokenizerFast, BertTokenizer, \
    TFBertForSequenceClassification, default_data_collator
from transformers import BertForSequenceClassification

from transformers import DataCollatorForLanguageModeling
from transformers import TFTrainer, TFTrainingArguments
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, Features, ClassLabel, Value, concatenate_datasets
from tqdm import tqdm
import logging
import random
import argparse
import json
import torch

from models.EnsembleAAEModel import EnsembleAAESpecialized
from tokenizer.tokenizers import get_aae_tokenizer
from utils.utils import get_line_by_line_ds, get_data_collator_for_lm, compute_metrics_trainer, show_gpu, \
    strip_punc_hp, compute_basic_metrics, plot_cm, compute_metrics_ensemble, compute_fairness_metrics

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--is_local",
                        action='store_true',
                        help="Is it being run locally")
    parser.add_argument("--is_gcloud",
                        action='store_true',
                        help="is it run in colab or pandas server")
    parser.add_argument("--is_gridsearch",
                        action='store_true',
                        help="Is gridsearch")
    parser.add_argument("--retrain",
                        action='store_true',
                        help="If directory exists, do you want to retrain")
    parser.add_argument("--task_name",
                        default="'mlm",
                        type=str,
                        required=True,
                        choices=['mlm', 'aae_ft', 'general_bert', 'ensemble'],
                        help="The name of the task to train.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        choices=['hate', 'harassment', 'davidson', 'founta', 'golbeck'])
    parser.add_argument('--batch_size_clsfr',
                        type=int,
                        default=32,
                        help="batch size for specialized classifier model training")
    parser.add_argument('--epochs_clsfr',
                        type=int,
                        default=1000,
                        help="epochs to train model specialized classifier")
    parser.add_argument('--learning_rate_clsfr',
                        type=float,
                        default=2e-5,
                        help="learning_rate for specialized classifier")
    parser.add_argument('--batch_size_lm',
                        type=int,
                        default=32,
                        help="batch size for language model training")
    parser.add_argument('--epochs_lm',
                        type=int,
                        default=1000,
                        help="epochs to lm train model")
    parser.add_argument('--learning_rate_lm',
                        type=float,
                        default=2e-5,
                        help="learning_rate for lm model")
    parser.add_argument('--max_seq_length',
                        type=int,
                        default=128,
                        help="max sequence length for embeddings")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    # if true, use test data instead of val data
    parser.add_argument("--test", action='store_true')

    args = parser.parse_args()

    LEARNING_RATE_CLSFR = args.learning_rate_clsfr
    LEARNING_RATE_MLM = args.learning_rate_lm
    BATCH_SIZE_CLSFR = args.batch_size_clsfr
    BATCH_SIZE_MLM = args.batch_size_lm
    EPOCHS_CLSFR = args.epochs_clsfr
    EPOCHS_MLM = args.epochs_lm
    MAX_SEQ_LENGTH = args.max_seq_length
    SEED = args.seed

    dataset_name = args.dataset.lower()
    task_name = args.task_name.lower()

    np.random.seed(SEED)
    torch.random.manual_seed(SEED)
    random.seed(SEED)

    is_local = args.is_local
    is_gcloud = args.is_gcloud
    local_data_prefix = '../../data/twitter_datasets/'
    cloud_root = 'drive/MyDrive/HateSpeech/benchmarking' if is_gcloud else '/nethome/mhalevy3/HateSpeech/benchmarking'
    root = cloud_root  # todo fix local file struct
    cloud_data_path_prefix = f'{root}/data/twitter/'

    data_prefix = local_data_prefix if is_local else cloud_data_path_prefix

    bert_uncased_tokenizer_path = f"{root}/models/bert_base_uncased_tokenizer/"
    blodgett_cleaned_aae_save_path = f'{root}/data/blodgett_aae_cleaned.txt'
    aae_tokenizer_path = f'{root}/models/aae_tokenizer/'

    # tuple of folder name containing the dev/test/training dataset, the label, and the text feature
    data_aae_map = {
        'hate': ('hate_aae/', 'is_hate', 'cleaned_tweet'),
        'harassment': ('harassment_aae/', 'is_harassment', 'cleaned_tweet'),
        'davidson': ('davidson_aae/', 'is_harassment', 'cleaned_tweet'),
        'founta': ('founta_aae/', 'is_harassment', 'cleaned_tweet'),
        'golbeck': ('golbeck_aae/', 'is_harassment', 'cleaned_tweet'),
    }
    data_map = {
        'hate': ('combined_hate/', 'is_hate', 'cleaned_tweet'),
        'harassment': ('combined_harassment/', 'is_harassment', 'cleaned_tweet'),
        'davidson': ('davidson/', 'is_harassment', 'cleaned_tweet'),
        'founta': ('founta/', 'is_harassment', 'cleaned_tweet'),
        'golbeck': ('golbeck/', 'is_harassment', 'cleaned_tweet'),
    }
    data_folder_aae, LABEL, TEXT_FEATURE = data_aae_map.get(dataset_name)
    data_folder, _, _ = data_map.get(dataset_name)
    data_path_aae = data_prefix + data_folder_aae
    data_path = data_prefix + data_folder

    affix_mlm = f'aaeMLM_LR_{strip_punc_hp(LEARNING_RATE_MLM)}_BS_{str(BATCH_SIZE_MLM)}_EP_{str(EPOCHS_MLM)}_MSL_{str(MAX_SEQ_LENGTH)}'
    affix_clsfr = f'{dataset_name}_LR_{strip_punc_hp(LEARNING_RATE_CLSFR)}_BS_{str(BATCH_SIZE_CLSFR)}_EP_{str(EPOCHS_CLSFR)}_aaeMLM_LR_{strip_punc_hp(LEARNING_RATE_MLM)}_BS_{str(BATCH_SIZE_MLM)}_EP_{str(EPOCHS_MLM)}_MSL_{str(MAX_SEQ_LENGTH)}'

    OUTPUT_DIR_MLM = f'{args.output_dir}/{affix_mlm}' if args.is_gridsearch else f'{args.output_dir}/mlm_{dataset_name}_{args.seed}'
    OUTPUT_DIR_AAE_CLSFR = f'{args.output_dir}/{affix_clsfr}' if args.is_gridsearch else f'{args.output_dir}/AAE_BERT_{dataset_name}_{args.seed}'
    OUTPUT_DIR_BERT = f'{args.output_dir}/BERT_{dataset_name}_{args.seed}'
    OUTPUT_DIR_ENSEMBLE = f'{args.output_dir}/ENSEMBLE_{dataset_name}_{args.seed}'

    output_map = {
        'aae_ft': OUTPUT_DIR_AAE_CLSFR,
        'mlm': OUTPUT_DIR_MLM,
        'general_bert': OUTPUT_DIR_BERT,
        'ensemble': OUTPUT_DIR_ENSEMBLE,
    }

    OUTPUT_DIR = output_map.get(task_name)

    # save configs

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    f = open(os.path.join(OUTPUT_DIR, 'args.json'), 'w')
    json.dump(args.__dict__, f, indent=4)
    f.close()

    def eval_ensemble_aae_model(learning_rate_spec=LEARNING_RATE_CLSFR,
                                batch_size_spec=BATCH_SIZE_CLSFR,
                                epochs_spec=EPOCHS_CLSFR,
                                learning_rate_MLM=LEARNING_RATE_MLM,
                                batch_size_MLM=BATCH_SIZE_MLM,
                                epochs_MLM=EPOCHS_MLM,
                                ):
        # should_retrain = args.retrain or not os.path.exists(f'{OUTPUT_DIR}/pytorch_model.bin')
        should_retrain = args.retrain or not os.path.exists(f'{OUTPUT_DIR}/pytorch_model.bin')

        if args.do_train and should_retrain:
        # training for the base models is done within their respective methods, this method is to load + eval
        # it can be modified to train the entire ensemble at once though.
            if not os.path.exists(OUTPUT_DIR_BERT):
                os.makedirs(OUTPUT_DIR_BERT)
            if not os.path.exists(OUTPUT_DIR_AAE_CLSFR):
                os.makedirs(OUTPUT_DIR_AAE_CLSFR)

            if not os.path.exists(f'{OUTPUT_DIR_BERT}/pytorch_model.bin'):
                logger.info('training bert classifier')
                eval_general_clsfr()
                show_gpu(f'GPU memory usage before clearing cache:')
                gc.collect()
                torch.cuda.empty_cache()
                show_gpu(f'GPU memory usage after clearing cache:')

            if not os.path.exists(f'{OUTPUT_DIR_AAE_CLSFR}/pytorch_model.bin'):
                logger.info('training specialized classificatier for aae')
                eval_aae_clsfr(learning_rate=learning_rate_spec,
                               batch_size=batch_size_spec,
                               epochs=epochs_spec,
                               learning_rate_mlm=learning_rate_MLM,
                               batch_size_mlm=batch_size_MLM,
                               epochs_mlm=epochs_MLM,
                               )

                show_gpu(f'GPU memory usage before clearing cache:')
                gc.collect()
                torch.cuda.empty_cache()
                show_gpu(f'GPU memory usage after clearing cache:')

        general_model = BertForSequenceClassification.from_pretrained(OUTPUT_DIR_BERT)
        aae_classif = BertForSequenceClassification.from_pretrained(OUTPUT_DIR_AAE_CLSFR)
        # can also be loaded but doesn't matter...
        ensemble = EnsembleAAESpecialized(general_model=general_model,
                                          specialized_model=aae_classif,
                                          )
        # ensemble.save_model(OUTPUT_DIR_ENSEMBLE) Not yet implemented

        general_model.eval()
        aae_classif.eval()

        aae_tokenizer = get_aae_tokenizer(
            aae_tokenizer_path,
            bert_uncased_tokenizer_path,
            blodgett_cleaned_aae_save_path,
            MAX_SEQ_LENGTH
        )

        sae_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased',
                                                          max_len=MAX_SEQ_LENGTH,
                                                          )
        dataset = load_dataset(
            'csv',
            data_files={
                'train': f'{data_path}/train.csv',
                'dev': f'{data_path}/dev.csv',
                'test': f'{data_path}/test.csv',
            },
            features=Features(
                {TEXT_FEATURE: Value('string'), LABEL: ClassLabel(names=['0', '1']), 'is_aae_06': Value('float')}),
        )

        def tokenize_function_aae(examples):
            return aae_tokenizer(examples[TEXT_FEATURE], padding="max_length", truncation=True)

        def tokenize_function_sae(examples):
            return sae_tokenizer(examples[TEXT_FEATURE], padding="max_length", truncation=True)

        aae_tokenized_datasets = dataset.map(tokenize_function_aae)
        sae_tokenized_datasets = dataset.map(tokenize_function_sae)

        aae_tokenized_datasets.rename_column_(original_column_name='attention_mask',
                                              new_column_name='attention_mask_aae')
        aae_tokenized_datasets.rename_column_(original_column_name='input_ids', new_column_name='input_ids_aae')
        aae_tokenized_datasets.rename_column_(original_column_name='token_type_ids',
                                              new_column_name='token_type_ids_aae')

        sae_tokenized_datasets.rename_column_(original_column_name='attention_mask',
                                              new_column_name='attention_mask_sae')
        sae_tokenized_datasets.rename_column_(original_column_name='input_ids', new_column_name='input_ids_sae')
        sae_tokenized_datasets.rename_column_(original_column_name='token_type_ids',
                                              new_column_name='token_type_ids_sae')
        sae_tokenized_datasets.rename_column_(original_column_name=LABEL, new_column_name='labels')

        aae_tokenized_datasets = aae_tokenized_datasets.remove_columns([LABEL, 'cleaned_tweet', 'is_aae_06'])
        sae_tokenized_datasets = sae_tokenized_datasets.remove_columns(['cleaned_tweet'])

        aae_train_ds = aae_tokenized_datasets['train']
        aae_dev_ds = aae_tokenized_datasets['dev']
        aae_test_ds = aae_tokenized_datasets['test']

        sae_train_ds = sae_tokenized_datasets['train']
        sae_dev_ds = sae_tokenized_datasets['dev']
        sae_test_ds = sae_tokenized_datasets['test']

        master_ds_tr = concatenate_datasets(dsets=[aae_train_ds, sae_train_ds], axis=1)
        master_ds_dev = concatenate_datasets(dsets=[aae_dev_ds, sae_dev_ds], axis=1)
        master_ds_test = concatenate_datasets(dsets=[aae_test_ds, sae_test_ds], axis=1)

        eval_ds = master_ds_test if args.test else master_ds_dev

        validate_w_fairness(model=ensemble, eval_ds=eval_ds, output_dir=OUTPUT_DIR_ENSEMBLE)

    def eval_aae_mlm(learning_rate=LEARNING_RATE_MLM,
                     batch_size=BATCH_SIZE_MLM,
                     epochs=EPOCHS_MLM,
                     ):
        should_retrain = args.retrain or not os.path.exists(f'{OUTPUT_DIR_MLM}/pytorch_model.bin')

        if args.do_train and should_retrain:
            aae_tokenizer = get_aae_tokenizer(
                aae_tokenizer_path,
                bert_uncased_tokenizer_path,
                blodgett_cleaned_aae_save_path,
                MAX_SEQ_LENGTH
            )

            dataset = get_line_by_line_ds(txt_path=blodgett_cleaned_aae_save_path,
                                          tokenizer=aae_tokenizer,
                                          max_seq_length=MAX_SEQ_LENGTH)

            data_collator = get_data_collator_for_lm(tokenizer=aae_tokenizer, mlm_prob=0.15)

            # Start from the weights of pre-trained BERT
            bert_base_lm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

            if not os.path.exists(OUTPUT_DIR_MLM):
                os.makedirs(OUTPUT_DIR_MLM)

            aae_lm_training_args = TrainingArguments(
                output_dir=OUTPUT_DIR_MLM,
                overwrite_output_dir=False,
                num_train_epochs=epochs,
                learning_rate=learning_rate,  # the default value
                per_gpu_train_batch_size=batch_size,
                save_steps=10_000,
                save_total_limit=2,
                prediction_loss_only=True,
                seed=SEED,
            )

            aae_lm_trainer = Trainer(
                model=bert_base_lm_model,
                args=aae_lm_training_args,
                data_collator=data_collator,
                train_dataset=dataset,
            )

            aae_lm_trainer.train()
            aae_lm_trainer.save_model(OUTPUT_DIR_MLM)

    def eval_general_clsfr():
        dataset = load_dataset(
            'csv',
            data_files={
                'train': f'{data_path}/train.csv',
                'dev': f'{data_path}/dev.csv',
                'test': f'{data_path}/test.csv'
            },
            features=Features({TEXT_FEATURE: Value('string'), LABEL: ClassLabel(names=['0', '1']), 'is_aae_06': Value('float')}),
        )

        bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', max_len=MAX_SEQ_LENGTH)

        def tokenize_function(examples):
            return bert_tokenizer(examples[TEXT_FEATURE], padding="max_length", truncation=True)

        tokenized_datasets = dataset.map(tokenize_function)

        train_dataset = tokenized_datasets['train']
        val_dataset = tokenized_datasets['dev']
        test_dataset = tokenized_datasets['test']

        train_dataset

        train_dataset.rename_column_(original_column_name=LABEL, new_column_name='label')
        val_dataset.rename_column_(original_column_name=LABEL, new_column_name='label')
        test_dataset.rename_column_(original_column_name=LABEL, new_column_name='label')

        should_retrain = args.retrain or not os.path.exists(f'{OUTPUT_DIR_BERT}/pytorch_model.bin')

        if not os.path.exists(OUTPUT_DIR_BERT):
            os.makedirs(OUTPUT_DIR_BERT)

        if args.do_train and should_retrain:
            bert_classifier = BertForSequenceClassification.from_pretrained('bert-base-uncased')

            params_map = {
                'hate': (3, 64, 2e-5),
                'harassment': (1, 32, 5e-5),
                'davidson': (5, 64, 5e-5),
                'founta': (1, 64, 2e-5),
                'golbeck': (3, 64, 2e-5),
            }

            epochs, batch_size, learning_rate = params_map.get(dataset_name)

            bert_classif_training_args_class = TrainingArguments(
                output_dir=OUTPUT_DIR_BERT,
                overwrite_output_dir=False,
                num_train_epochs=epochs,
                learning_rate=learning_rate,  # the default value
                per_device_train_batch_size=16,
                per_device_eval_batch_size=64,
                save_steps=10_000,
                save_total_limit=2,
                warmup_steps=500,  # number of warmup steps for learning rate scheduler
                weight_decay=0.01,
                prediction_loss_only=True,
                seed=SEED,
            )

            bert_classif_trainer = Trainer(
                model=bert_classifier,
                args=bert_classif_training_args_class,
                compute_metrics=compute_metrics_trainer,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
            )

            bert_classif_trainer.train()
            bert_classif_trainer.save_model(OUTPUT_DIR_BERT)
        else:
            bert_classif_trainer = BertForSequenceClassification.from_pretrained(OUTPUT_DIR_BERT)

        eval_ds = test_dataset if args.test else val_dataset

        validate_w_fairness(model=bert_classif_trainer, eval_ds=eval_ds, output_dir=OUTPUT_DIR_BERT)

    def eval_aae_clsfr(learning_rate=LEARNING_RATE_CLSFR,
                       batch_size=BATCH_SIZE_CLSFR,
                       epochs=EPOCHS_CLSFR,
                       learning_rate_mlm=LEARNING_RATE_MLM,
                       batch_size_mlm=BATCH_SIZE_MLM,
                       epochs_mlm=EPOCHS_MLM
                       ):

        dataset = load_dataset(
            'csv',
            data_files={
                'train': f'{data_path_aae}/train.csv',
                'dev': f'{data_path_aae}/dev.csv',
                'test': f'{data_path_aae}/test.csv'
            },
            features=Features({TEXT_FEATURE: Value('string'), LABEL: ClassLabel(names=['0', '1'])}),
        )

        aae_tokenizer = get_aae_tokenizer(
            aae_tokenizer_path,
            bert_uncased_tokenizer_path,
            blodgett_cleaned_aae_save_path,
            MAX_SEQ_LENGTH
        )

        def tokenize_function(examples):
            return aae_tokenizer(examples[TEXT_FEATURE], padding="max_length", truncation=True)

        tokenized_datasets = dataset.map(tokenize_function)

        train_dataset = tokenized_datasets['train']
        val_dataset = tokenized_datasets['dev']
        test_dataset = tokenized_datasets['test']

        train_dataset.rename_column_(original_column_name=LABEL, new_column_name='label')
        val_dataset.rename_column_(original_column_name=LABEL, new_column_name='label')
        test_dataset.rename_column_(original_column_name=LABEL, new_column_name='label')

        should_retrain = args.retrain or not os.path.exists(f'{OUTPUT_DIR_AAE_CLSFR}/pytorch_model.bin')

        if not os.path.exists(OUTPUT_DIR_AAE_CLSFR):
            os.makedirs(OUTPUT_DIR_AAE_CLSFR)

        if args.do_train and should_retrain:
            # load fine-tuned AAE Bert LM, if it doesn't exist train it
            if not os.path.exists(f'{OUTPUT_DIR_MLM}/pytorch_model.bin'):
                logger.info('training mlm for aae classifaction')
                eval_aae_mlm(learning_rate=learning_rate_mlm,
                             batch_size=batch_size_mlm,
                             epochs=epochs_mlm,
                             )

                show_gpu(f'GPU memory usage before clearing cache:')
                gc.collect()
                torch.cuda.empty_cache()
                show_gpu(f'GPU memory usage after clearing cache:')

            bertAAEClassifier = BertForSequenceClassification.from_pretrained(OUTPUT_DIR_MLM)

            aae_classif_training_args_class = TrainingArguments(
                output_dir=OUTPUT_DIR_AAE_CLSFR,
                overwrite_output_dir=True,
                num_train_epochs=epochs,
                learning_rate=learning_rate,  # the default value
                per_device_train_batch_size=batch_size,
                save_steps=10_000,
                save_total_limit=2,
                seed=SEED,
            )

            aae_classif_trainer = Trainer(
                model=bertAAEClassifier,
                args=aae_classif_training_args_class,
                compute_metrics=compute_metrics_trainer,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
            )

            aae_classif_trainer.train()
            aae_classif_trainer.save_model(OUTPUT_DIR_AAE_CLSFR)

            # aae_classif_trainer.predict(test_dataset)
        else:
            aae_classif_trainer = BertForSequenceClassification.from_pretrained(OUTPUT_DIR_AAE_CLSFR)

        eval_ds = test_dataset if args.test else val_dataset

        validate_no_fairness(model=aae_classif_trainer, eval_ds=eval_ds, output_dir=OUTPUT_DIR)


    def validate_w_fairness(model, eval_ds, output_dir=OUTPUT_DIR):
        split = 'dev' if not args.test else 'test'

        if type(model) is Trainer:
            result = model.evaluate()
            pred = model.predict(eval_ds)
            result = compute_fairness_metrics(result,
                                              pred.predictions.argmax(-1),
                                              eval_ds['is_aae_06'],
                                              pred.label_ids,
                                              )
            plot_cm(output_dir=output_dir,
                    task_name=task_name,
                    labels=pred.label_ids,
                    predictions=pred.predictions.argmax(-1),
                    train_dev_test=split,
                    )

        else:
            if type(model) is BertForSequenceClassification:
                eval_ds.rename_column_(original_column_name='label', new_column_name='labels')
                labels = eval_ds['labels']
                is_aae = eval_ds['is_aae_06']
                eval_ds.set_format(type='pt', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
                model.eval()

                dataloader_eval = torch.utils.data.DataLoader(eval_ds, batch_size=64)

            elif type(model) is EnsembleAAESpecialized:
                labels = eval_ds['labels']
                is_aae = eval_ds['is_aae_06']
                eval_ds.set_format(type='pt', columns=['attention_mask_aae', 'input_ids_aae', 'token_type_ids_aae',
                                                            'attention_mask_sae', 'input_ids_sae', 'is_aae_06',
                                                            'token_type_ids_sae'])

                dataloader_eval = torch.utils.data.DataLoader(eval_ds, batch_size=64)

            preds = []
            pred_prs = []

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.eval()
            model.to(device)
            with torch.no_grad():
                for i, batch in enumerate(tqdm(dataloader_eval)):
                    batch = {k: v.to(device) for k, v in batch.items()}

                    output = model(**batch)
                    pred_pr = torch.sigmoid(output)
                    pred = torch.argmax(pred_pr, 1)
                    pred = pred.cpu().numpy()
                    pred_pr = pred_pr.cpu().numpy()[:, 1]
                    # losses.append(output['loss'].cpu())
                    preds.append(pred)
                    pred_prs.append(pred_pr)

            preds = np.concatenate(preds).ravel()
            preds_prs = np.concatenate(pred_prs).ravel()

            result = compute_metrics_ensemble(labels=labels,
                                              preds=preds,
                                              preds_prob=preds_prs,
                                              in_group_labels=is_aae)
            plot_cm(output_dir, task_name, labels, preds, split)

        output_eval_file = os.path.join(output_dir, f"eval_results_{split}_{args.task_name}")
        with open(output_eval_file, "w", encoding="utf-8") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))



    # assumes model is either Trainer or BertForSequenceClassification, eval_ds is of HuggingFace datasets type thats
    # already tokenized
    def validate_no_fairness(model, eval_ds, output_dir=OUTPUT_DIR):
        split = 'dev' if not args.test else 'test'

        if type(model) is Trainer:
            result = model.evaluate()
            pred = model.predict(eval_ds)
            plot_cm(output_dir=OUTPUT_DIR,
                    task_name=task_name,
                    labels=pred.label_ids,
                    predictions=pred.predictions.argmax(-1),
                    train_dev_test=split)

        elif type(model) is BertForSequenceClassification:
            eval_ds.set_format(type='pt', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
            model.eval()
            input_ids = eval_ds['input_ids']
            attention_mask = eval_ds['attention_mask']
            token_type_ids = eval_ds['token_type_ids']
            # move everything to cuda
            model.to('cuda')

            input_ids = input_ids.to('cuda')
            attention_mask = attention_mask.to('cuda')
            token_type_ids = token_type_ids.to('cuda')
            with torch.no_grad():
                output = model(input_ids, attention_mask, token_type_ids)
                pred_pr = torch.sigmoid(output['logits'])
                pred = torch.argmax(pred_pr, 1)
                pred = pred.cpu().numpy()
                pred_pr = pred_pr.cpu().numpy()[:, 1]

            result = compute_basic_metrics(eval_ds['label'], pred, pred_pr)

            plot_cm(OUTPUT_DIR, task_name, eval_ds['label'], pred, split)

        output_eval_file = os.path.join(output_dir, f"eval_results_{split}_{args.task_name}")
        with open(output_eval_file, "w", encoding="utf-8") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    params = {
        'mlm': None,
        'aae_ft': None,
        'general_bert': None,
        'ensemble': None,
    }

    kwargs = params.get(task_name, None)

    if task_name == 'mlm':
        eval_aae_mlm()
    elif task_name == 'aae_ft':
        eval_aae_clsfr()
    elif task_name == 'general_bert':
        eval_general_clsfr()
    elif task_name == 'ensemble':
        eval_ensemble_aae_model()

    show_gpu(f'GPU memory usage before clearing cache:')
    gc.collect()
    torch.cuda.empty_cache()
    show_gpu(f'GPU memory usage after clearing cache:')


if __name__ == "__main__":
    main()
