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
# from transformers import BertTokenizer, glue_convert_examples_to_features
# from transformers import TFBertForSequenceClassification
# import tfrecorder #https://github.com/google/tensorflow-recorder

from src.utils.utils import tfidf_vectorize, logistic_regression_model, compute_metrics, strip_punc_hp, \
    f1_from_prec_recall, compute_disparate_impact, remove_punctuation_tweet, glove_vectorize, train_plot

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--is_local",
                        default=True,
                        type=bool,
                        required=True,
                        help="Is it being run in colab or locally")
    parser.add_argument("--task_name",
                        default="ngram",
                        type=str,
                        required=True,
                        choices=['ngram', 'tf_idf', 'glove', 'bert', 'bert_mindiff'],
                        help="The name of the task to train.")  # todo set up
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
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

    is_local = args.is_local  # todo
    local_data_path = '../../data/twitter_datasets/combined_harassment/'
    gdrive_data_path = 'drive/My\ Drive/Hate\ Speech\ Research/contextualizing-hate-speech-models-with-explanations-master/data/twitter/combined_harassment/'

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # save configs
    f = open(os.path.join(args.output_dir, 'args.json'), 'w')
    json.dump(args.__dict__, f, indent=4)
    f.close()

    task_name = args.task_name.lower()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    random.seed(args.seed)

    data_path = local_data_path if is_local else gdrive_data_path

    dev_pd = pd.read_csv(f'{data_path}dev.csv', index_col=None).dropna()
    train_pd = pd.read_csv(f'{data_path}train.csv', index_col=None).dropna()
    test_pd = pd.read_csv(f'{data_path}test.csv', index_col=None).dropna()

    LABEL = 'is_harassment'
    TEXT_FEATURE = 'cleaned_tweet'
    BATCH_SIZE = 512

    def eval_ngram_logreg(train_data,
                          validation_data,
                          test_data,
                          learning_rate=1e-3,
                          epochs=1000,  # todo change
                          batch_size=512,
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
            # todo https://towardsdatascience.com/implementing-macro-f1-score-in-keras-what-not-to-do-e9f1aa04029d
            model.compile(loss='bce',
                          optimizer=optimizer,
                          metrics=['acc', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                                   tf.keras.metrics.AUC()]
                          )
            # early stopping if validation loss does not decrease in 2 consecutive tries.
            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
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

            # Save model.
            model.save(f'{args.output_dir}/{task_name}_logreg.h5')
        else:
            model = tf.keras.models.load_model(f'{args.output_dir}/{task_name}_logreg.h5')

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
                          learning_rate=1e-3,
                          epochs=1000,  # todo change
                          batch_size=512):
        # todo change to local vs cloud
        path_to_glove_file = '../../models/glove/glove.6B.100d.txt'

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

            #todo: https://github.com/keras-team/keras/issues/11749
            # try with custom metrics on output instead like f1
            model.compile(loss='bce',
                          optimizer=optimizer,
                          metrics=['acc'] # , tf.keras.metrics.Precision()] #, tf.keras.metrics.Recall()],
                                   #tf.keras.metrics.AUC()]
                          )
            # early stopping if validation loss does not decrease in 2 consecutive tries.
            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
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
            train_plot(history, args.output_dir, task_name)

            # Save model.
            model.save(f'{args.output_dir}/{task_name}_logreg.h5')
        else:
            model = tf.keras.models.load_model(f'{args.output_dir}/{task_name}_logreg.h5')

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

        eval = model.evaluate(eval_examples, eval_labels)
        y_hat_test = np.argmax(model.predict(eval_examples), axis=-1)

        # corresponds with metrics in model.compile
        metrics_list = ["loss", "acc", "precision", "recall", "auc_roc"]
        result = dict(zip(metrics_list, eval))
        if task_name != 'glove':
            #todo
            result['f1'] = f1_from_prec_recall(result['precision'], result['recall'])
        result = compute_disparate_impact(metrics_dict=result,
                                          preds=y_hat_test,
                                          in_group_labels_08=in_group_labels_08,
                                          in_group_labels_06=in_group_labels_06,
                                          )
        result['loss'] = eval[0]
        split = 'dev' if not args.test else 'test'

        output_eval_file = os.path.join(args.output_dir, f"eval_results_{split}_{args.task_name}")
        with open(output_eval_file, "w", encoding="utf-8") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        # test_acc = compute_metrics(y_hat_test == y_test)
        # print(f'Test accuracy: {test_acc}')

    params = {
        'ngram': {'train_data': train_pd, 'validation_data': dev_pd, 'test_data': test_pd, 'tf_idf': False,
                  'ngram_range': (1, 1)},
        'tf_idf': {'train_data': train_pd, 'validation_data': dev_pd, 'test_data': test_pd, 'tf_idf': True,
                   'ngram_range': (2, 2)},
        'glove': {'train_data': train_pd, 'validation_data': dev_pd, 'test_data': test_pd},
        'bert': None,
        'bert_mindiff': None,
    }

    if task_name == 'ngram' or task_name == 'tf_idf':
        kwargs = params.get(task_name, None)
        eval_ngram_logreg(**kwargs)
    elif task_name == 'glove':
        kwargs = params.get(task_name, None)
        eval_glove_logreg(**kwargs)




if __name__ == "__main__":
    main()
