import copy
import os
import requests
import tempfile
import urllib
import zipfile
import string
import re

import tensorflow_model_remediation.min_diff as md
from google.protobuf import text_format
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_model_analysis as tfma
import tensorflow_data_validation as tfdv
from tensorflow_model_analysis.addons.fairness.post_export_metrics import fairness_indicators
from tensorflow_model_analysis.addons.fairness.view import widget_view
import sys
import os
import logging
import random
import argparse
import sys
import json

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from tensorflow.python.keras import models
from transformers import BertTokenizer
from transformers import TFBertForSequenceClassification
# import tfrecorder #https://github.com/google/tensorflow-recorder

from src.utils.utils import tfidf_vectorize, logistic_regression_model, compute_metrics, strip_punc_hp, \
    f1_from_prec_recall, compute_fairness_metrics, remove_punctuation_tweet, glove_vectorize, train_plot, \
    prepare_mindiff_ds

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--is_local",
                        default=True,
                        type=bool,
                        required=True,
                        help="Is it being run in colab or locally")
    parser.add_argument("--is_gridsearch",
                        default=False,
                        type=bool,
                        required=True,
                        help="Is it being run in colab or locally")
    parser.add_argument("--task_name",
                        default="ngram",
                        type=str,
                        required=True,
                        choices=['ngram', 'tf_idf', 'glove', 'bert', 'bert_mindiff'],
                        help="The name of the task to train.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        choices=['hate', 'harassment', 'davidson', 'founta', 'waseem', 'golbeck'])
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help="batch size for model training")
    parser.add_argument('--epochs',
                        type=int,
                        default=1000,
                        help="epochs to train model")
    parser.add_argument('--max_seq_length',
                        type=int,
                        default=128,
                        help="max sequence length for embeddings")
    parser.add_argument('--learning_rate',
                        type=float,
                        default=2e-5,
                        help="learning_rate")
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

    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    MAX_SEQ_LENGTH = args.max_seq_length

    dataset_name = args.dataset.lower()
    task_name = args.task_name.lower()

    affix = f'{dataset_name}_LR_{strip_punc_hp(LEARNING_RATE)}_BS_{str(BATCH_SIZE)}_EP_{str(EPOCHS)}_MSL_{str(MAX_SEQ_LENGTH)}'

    OUTPUT_DIR = f'{args.output_dir}/{affix}' if args.is_gridsearch else f'{args.output_dir}/{dataset_name}'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # save configs
    f = open(os.path.join(OUTPUT_DIR, 'args.json'), 'w')
    json.dump(args.__dict__, f, indent=4)
    f.close()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)

    is_local = args.is_local
    local_data_prefix = '../../data/twitter_datasets/'
    gdrive_data_path_prefix = 'drive/My\ Drive/HateSpeech/benchmarking/data/twitter/'

    data_prefix = local_data_prefix if is_local else gdrive_data_path_prefix

    # tuple of folder name containing the dev/test/training dataset, the label, and the text feature
    data_map = {
        'hate': ('combined_hate/', 'is_hate', 'cleaned_tweet'),
        'harassment': ('combined_harassment/', 'is_harassment', 'cleaned_tweet'),
        'davidson': ('davidson/', 'is_harassment', 'cleaned_tweet'),
        'founta': ('founta/', 'is_harassment', 'cleaned_tweet'),
        'waseem': ('waseem/', 'is_hate', 'cleaned_tweet'),
        'golbeck': ('golbeck/', 'is_harassment', 'cleaned_tweet'),
    }
    data_folder, LABEL, TEXT_FEATURE = data_map.get(dataset_name, sys.stderr.write('incorrect dataset type chosen'))
    data_path = data_prefix + data_folder

    dev_pd = pd.read_csv(f'{data_path}dev.csv', index_col=None).dropna()
    train_pd = pd.read_csv(f'{data_path}train.csv', index_col=None).dropna()
    test_pd = pd.read_csv(f'{data_path}test.csv', index_col=None).dropna()

    def eval_min_diff_bert(train_data,
                           validation_data,
                           test_data,
                           learning_rate=LEARNING_RATE,
                           batch_size=BATCH_SIZE,
                           epochs=EPOCHS,
                           path_to_bert=f'../existing_models/contextual-hsd-expl/runs/twitter_harass_es_vanilla_bal_seed_{args.seed}',
                           ):

        train_ds_main, train_ds_unpriv, train_ds_priv, (dev_examples, y_dev), (
            test_examples, y_test) = prepare_mindiff_ds(train_data=train_data,
                                                        validation_data=validation_data,
                                                        test_data=test_data,
                                                        unpriv_label='is_aae_06',
                                                        text_feature=TEXT_FEATURE,
                                                        label=LABEL,
                                                        batch_size=batch_size,
                                                        max_seq_length=MAX_SEQ_LENGTH)

        dev_ds = tf.data.Dataset.from_tensor_slices((dev_examples,
                                                     y_dev.reshape(-1, 1) * 1.0)
                                                    ).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((test_examples,
                                                      y_test.reshape(-1, 1) * 1.0)
                                                     ).batch(batch_size)
        validate_tfrecord_file = test_ds if args.test else dev_ds

        if args.do_train:
            bert_model = TFBertForSequenceClassification.from_pretrained(path_to_bert, from_pt=True)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            loss = tf.keras.losses.BinaryCrossentropy()
            bert_model.compile(optimizer=optimizer, loss=loss)

            min_diff_weight = 1.5  # @param {type:"number"}

            # Create the dataset that will be passed to the MinDiffModel during training.
            dataset = md.keras.utils.input_utils.pack_min_diff_data(
                train_ds_main, train_ds_unpriv, train_ds_priv
            )

            # Wrap the original model in a MinDiffModel, passing in one of the MinDiff
            # losses and using the set loss_weight.
            min_diff_loss = md.losses.MMDLoss()
            model = md.keras.MinDiffModel(bert_model,
                                          min_diff_loss,
                                          min_diff_weight)

            # Compile the model normally after wrapping the original model.  Note that
            # this means we use the baseline's model's loss here.
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            loss = tf.keras.losses.BinaryCrossentropy()
            # early stopping if validation loss does not decrease in 2 consecutive tries.
            callbacks = [
                # tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
                tf.keras.callbacks.TensorBoard(log_dir="logs")
            ]
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

            history = model.fit(dataset,
                                epochs=epochs,
                                callbacks=callbacks,
                                verbose=2,
                                batch_size=batch_size,
                                )

            history = history.history
            val_acc = history["val_acc"][-1]
            val_loss = history["val_loss"][-1]
            print(f'Validation accuracy: {val_acc}, loss: {val_loss}')
            train_plot(history, OUTPUT_DIR, task_name)

            model.save_original_model(f'{OUTPUT_DIR}/{task_name}', save_format='tf')
        else:
            model = tf.keras.models.load_model(f'{OUTPUT_DIR}/{task_name}')

        eval_examples, eval_labels = (test_examples, y_test) if args.test else (dev_examples, y_dev)
        in_group_06 = test_data['is_aae_06'].values if args.test else validation_data['is_aae_06'].values
        in_group_08 = test_data['is_aae_08'].values if args.test else validation_data['is_aae_08'].values

        validate(model=model,
                 eval_examples=eval_examples,
                 eval_labels=eval_labels,
                 in_group_labels_06=in_group_06,
                 in_group_labels_08=in_group_08)

    def eval_ngram_logreg(train_data,
                          validation_data,
                          test_data,
                          learning_rate=LEARNING_RATE,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE,
                          tf_idf=True,
                          ngram_range=(1, 2)):
        """

        :param train_data: pandas dataframe of the training data
        :param validation_data: pandas dataframe of the validation data
        :param test_data: pandas dataframe of the test data
        :param learning_rate: float, learning rate for training model.
        :param epochs: int, number of epochs.
        :param batch_size: int, number of samples per batch.
        :param tf_idf: bool, whether to encode tf-idf or n-gram
        :return:
        """
        train_texts, y_train = train_data[TEXT_FEATURE].values, train_data[LABEL].values
        dev_texts, y_dev = validation_data[TEXT_FEATURE].values, validation_data[LABEL].values
        test_texts, y_test = test_data[TEXT_FEATURE].values, test_data[LABEL].values

        x_train, x_dev, x_test = tfidf_vectorize(train_texts=train_texts,
                                                 train_labels=y_train,
                                                 val_texts=dev_texts,
                                                 test_texts=test_texts,
                                                 tf_idf=tf_idf,
                                                 ngram_range=ngram_range)
        if args.do_train:
            model = logistic_regression_model(x_train.shape[1])
            optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
            model.compile(loss='bce',
                          optimizer=optimizer,
                          metrics=['acc', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                                   tf.keras.metrics.AUC()]
                          )
            # early stopping if validation loss does not decrease in 2 consecutive tries.
            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
                tf.keras.callbacks.TensorBoard(log_dir="logs")
            ]

            ## tensorboard todo: explore
            ## % load_ext
            ## tensorboard
            ## % tensorboard - -logdir
            ## logs
            ##
            history = model.fit(x_train,
                                y_train,
                                epochs=epochs,
                                callbacks=callbacks,
                                validation_data=(x_dev, y_dev),
                                verbose=2,  # once per epoch
                                batch_size=batch_size,
                                )

            # Print results.
            history = history.history
            val_acc = history["val_acc"][-1]
            val_loss = history["val_loss"][-1]
            print(f'Validation accuracy: {val_acc}, loss: {val_loss}')
            train_plot(history, OUTPUT_DIR, task_name)

            # Save model.
            model.save(f'{OUTPUT_DIR}/{task_name}_logreg.h5')
        else:
            model = tf.keras.models.load_model(f'{OUTPUT_DIR}/{task_name}_logreg.h5')

        eval_examples, eval_labels = (x_test, y_test) if args.test else (x_dev, y_dev)
        in_group_06 = test_data['is_aae_06'].values if args.test else validation_data['is_aae_06'].values
        in_group_08 = test_data['is_aae_08'].values if args.test else validation_data['is_aae_08'].values

        validate(model=model,
                 eval_examples=eval_examples,
                 eval_labels=eval_labels,
                 in_group_labels_06=in_group_06,
                 in_group_labels_08=in_group_08)

    def eval_glove_logreg(train_data,
                          validation_data,
                          test_data,
                          learning_rate=LEARNING_RATE,
                          epochs=EPOCHS,
                          batch_size=BATCH_SIZE):
        local_glove_prefix = '../../models/glove/'
        gdrive_glove_prefix = 'drive/My\ Drive/HateSpeech/benchmarking/models/glove/'

        glove_prefix = local_glove_prefix if is_local else gdrive_glove_prefix
        path_to_glove_file = glove_prefix + 'glove.6B.100d.txt'

        train_texts, y_train = train_data[TEXT_FEATURE].values, train_data[LABEL].values
        dev_texts, y_dev = validation_data[TEXT_FEATURE].values, validation_data[LABEL].values
        test_texts, y_test = test_data[TEXT_FEATURE].values, test_data[LABEL].values

        x_train, x_dev, x_test, embedding_layer = glove_vectorize(train_texts=train_texts,
                                                                  val_texts=dev_texts,
                                                                  test_texts=test_texts,
                                                                  path_to_glove_file=path_to_glove_file,
                                                                  )

        if args.do_train:
            int_sequences_input = tf.keras.Input(shape=(None,), dtype="int64")
            # embedded_sequences = embedding_layer(int_sequences_input)
            model = logistic_regression_model(x_train.shape[1], embedding_layer)
            optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

            # todo: https://github.com/keras-team/keras/issues/11749
            # try with custom metrics on output instead like f1
            model.compile(loss='bce',
                          optimizer=optimizer,
                          metrics=['acc', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                                   tf.keras.metrics.AUC()]
                          )
            # early stopping if validation loss does not decrease in 5 consecutive tries.
            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
                tf.keras.callbacks.TensorBoard(log_dir="logs")
            ]

            history = model.fit(x_train,
                                y_train,
                                epochs=epochs,
                                callbacks=callbacks,
                                validation_data=(x_dev, y_dev),
                                verbose=2,  # once per epoch
                                batch_size=batch_size,
                                )

            # Print results.
            history = history.history
            val_acc = history["val_acc"][-1]
            val_loss = history["val_loss"][-1]
            print(f'Validation accuracy: {val_acc}, loss: {val_loss}')
            train_plot(history, OUTPUT_DIR, task_name)

            # Save model.
            model.save(f'{OUTPUT_DIR}/{task_name}_logreg.h5')
        else:
            model = tf.keras.models.load_model(f'{OUTPUT_DIR}/{task_name}_logreg.h5')

        eval_examples, eval_labels = (x_test, y_test) if args.test else (x_dev, y_dev)
        in_group_06 = test_data['is_aae_06'].values if args.test else validation_data['is_aae_06'].values
        in_group_08 = test_data['is_aae_08'].values if args.test else validation_data['is_aae_08'].values

        validate(model=model,
                 eval_examples=eval_examples,
                 eval_labels=eval_labels,
                 in_group_labels_06=in_group_06,
                 in_group_labels_08=in_group_08)

        # how to make end to end model :
        # string_input = tf.keras.Input(shape=(1,), dtype="string")
        # x = vectorizer(string_input)
        # preds = model(x)
        # end_to_end_model = keras.Model(string_input, preds)

        # probabilities = end_to_end_model.predict(
        #     [["this message is about computer graphics and 3D modeling"]]
        # )
        #
        # class_names[np.argmax(probabilities[0])]

    def validate(model, eval_examples, eval_labels, in_group_labels_06, in_group_labels_08):
        if task_name == 'bert_mindiff' or task_name == 'bert':
            model.eval()  # turns off the DropOut Module
        eval = model.evaluate(eval_examples, eval_labels)
        y_hat_test = np.argmax(model.predict(eval_examples), axis=-1)

        # corresponds with metrics in model.compile
        metrics_list = ["loss", "acc", "precision", "recall", "auc_roc"]
        result = dict(zip(metrics_list, eval))
        result['f1'] = f1_from_prec_recall(result['precision'], result['recall'])
        result = compute_fairness_metrics(metrics_dict=result,
                                          preds=y_hat_test,
                                          in_group_labels_08=in_group_labels_08,
                                          in_group_labels_06=in_group_labels_06,
                                          true_labels=eval_labels,
                                          )
        result['loss'] = eval[0]
        split = 'dev' if not args.test else 'test'

        output_eval_file = os.path.join(OUTPUT_DIR, f"eval_results_{split}_{args.task_name}")
        with open(output_eval_file, "w", encoding="utf-8") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    bert_local_prefix = '../existing_models/'
    bert_gdrive_prefix = 'drive/My\ Drive/HateSpeech/benchmarking/'
    bert_prefix = bert_local_prefix if is_local else bert_gdrive_prefix
    bert_path = f'{bert_prefix}contextual-hsd-expl/runs/twitter_harass_es_vanilla_bal_seed_{args.seed}'

    params = {
        'ngram': {'train_data': train_pd, 'validation_data': dev_pd, 'test_data': test_pd, 'tf_idf': False,
                  'ngram_range': (1, 1)},
        'tf_idf': {'train_data': train_pd, 'validation_data': dev_pd, 'test_data': test_pd, 'tf_idf': True,
                   'ngram_range': (2, 2)},
        'glove': {'train_data': train_pd, 'validation_data': dev_pd, 'test_data': test_pd},
        'bert': None,
        'bert_mindiff': {'train_data': train_pd, 'validation_data': dev_pd, 'test_data': test_pd,
                         'path_to_bert': bert_path},
    }

    kwargs = params.get(task_name, None)
    if task_name == 'ngram' or task_name == 'tf_idf':
        eval_ngram_logreg(**kwargs)
    elif task_name == 'glove':
        eval_glove_logreg(**kwargs)
    elif task_name == 'bert':
        pass
    elif task_name == 'bert_mindiff':
        eval_min_diff_bert(**kwargs)


if __name__ == "__main__":
    main()
