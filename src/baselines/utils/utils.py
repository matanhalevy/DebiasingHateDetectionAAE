import copy
from collections import defaultdict
# from time import clock
from functools import reduce
import math

from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils import compute_sample_weight
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from tensorflow.python.keras import models
import re
import string

from sklearn.model_selection import learning_curve, GridSearchCV
from tensorflow.python.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


# from transformers import BertTokenizer


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, pred_probs):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    f1_w = f1_score(y_true=labels, y_pred=preds, average='weighted')
    p, r = precision_score(y_true=labels, y_pred=preds), recall_score(y_true=labels, y_pred=preds)
    p_w, r_w = precision_score(y_true=labels, y_pred=preds, average='weighted'), recall_score(y_true=labels,
                                                                                              y_pred=preds,
                                                                                              average='weighted')
    try:
        roc = roc_auc_score(y_true=labels, y_score=pred_probs[:, 1])
    except ValueError:
        roc = 0.
    return {
        "acc": acc,
        "f1": f1,
        "precision": p,
        "recall": r,
        "auc_roc": roc,
        "precision_weighted": p_w,
        "recall_weighted": r_w,
        "f1_weighted": f1_w,
    }


def get_basic_metrics(labels, preds, pred_probs):
    prec, recall = precision_score(y_true=labels, y_pred=preds), recall_score(y_true=labels, y_pred=preds)
    try:
        auc_roc = roc_auc_score(y_true=labels, y_score=pred_probs)
    except ValueError:
        auc_roc = 0.

    return prec, recall, auc_roc

def f1_from_prec_recall(prec, recall):
    return 2 * safe_division((prec * recall), (prec + recall))


def compute_metrics(preds, labels, pred_probs, in_group_labels_08, in_group_labels_06):
    assert len(preds) == len(labels)
    metrics_dict = acc_and_f1(preds, labels, pred_probs)
    metrics_dict = compute_fairness_metrics(metrics_dict, preds, labels, pred_probs, in_group_labels_08,
                                            in_group_labels_06)
    return metrics_dict


def compute_metrics_custom(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def compute_fairness_metrics(metrics_dict, preds, in_group_labels_08, in_group_labels_06, true_labels):
    results_df = pd.DataFrame()
    results_df['pred'] = preds
    results_df['label'] = true_labels
    results_df['is_aae_08'] = in_group_labels_08
    results_df['is_aae_06'] = in_group_labels_06

    def favorable(series):
        favorable_ser = series[series == 0]
        return len(favorable_ser)

    def unfavorable(series):
        unfavorable_ser = series[series == 1]
        return len(unfavorable_ser)

    favorable_counts_df = results_df.groupby(by='is_aae_08').agg(
        {'pred': ['count', favorable, unfavorable]}).reset_index()

    if favorable_counts_df.shape == (2, 4):
        priv_total = favorable_counts_df.iloc[0, 1]
        unpriv_total = favorable_counts_df.iloc[1, 1]
        # favorable is outcome is not hate/harassment/abusive
        unpriv_ratio_favorable = safe_division(favorable_counts_df.iloc[1, 2], unpriv_total)
        priv_ratio_favorable = safe_division(favorable_counts_df.iloc[0, 2], priv_total)
        disparate_impact_favorable = safe_division(unpriv_ratio_favorable, priv_ratio_favorable)

        unpriv_ratio_unfavorable = safe_division(favorable_counts_df.iloc[1, 3], unpriv_total)
        priv_ratio_unfavorable = safe_division(favorable_counts_df.iloc[0, 3], priv_total)
        disparate_impact_unfavorable = safe_division(unpriv_ratio_unfavorable, priv_ratio_unfavorable)

        fpr_unpriv = safe_division(
            results_df[(results_df['is_aae_08'] == 1) & (results_df['pred'] == 1) & (results_df['label'] == 0)].shape[
                0],
            results_df[(results_df['is_aae_08'] == 1) & (results_df['label'] == 0)].shape[0]
        )
        fpr_priv = safe_division(
            results_df[(results_df['is_aae_08'] == 0) & (results_df['pred'] == 1) & (results_df['label'] == 0)].shape[
                0],
            results_df[(results_df['is_aae_08'] == 0) & (results_df['label'] == 0)].shape[0]
        )
        fpr_total = safe_division(
            results_df[(results_df['pred'] == 1) & (results_df['label'] == 0)].shape[0],
            results_df[(results_df['label'] == 0)].shape[0]
        )

        fnr_unpriv = safe_division(
            results_df[(results_df['is_aae_08'] == 1) & (results_df['pred'] == 0) & (results_df['label'] == 1)].shape[
                0],
            results_df[(results_df['is_aae_08'] == 1) & (results_df['label'] == 1)].shape[0]
        )
        fnr_priv = safe_division(
            results_df[(results_df['is_aae_08'] == 0) & (results_df['pred'] == 0) & (results_df['label'] == 1)].shape[
                0],
            results_df[(results_df['is_aae_08'] == 0) & (results_df['label'] == 1)].shape[0]
        )
        fnr_total = safe_division(
            results_df[(results_df['pred'] == 0) & (results_df['label'] == 1)].shape[0],
            results_df[(results_df['label'] == 1)].shape[0]
        )

        metrics_dict['unpriv_total_08'] = unpriv_total
        metrics_dict['priv_total_08'] = priv_total
        metrics_dict['fpr_unpriv_08'] = fpr_unpriv
        metrics_dict['fpr_priv_08'] = fpr_priv
        metrics_dict['fpr_total_08'] = fpr_total
        metrics_dict['fnr_unpriv_08'] = fnr_unpriv
        metrics_dict['fnr_priv_08'] = fnr_priv
        metrics_dict['fnr_total_08'] = fnr_total
        metrics_dict['disparate_impact_favorable_08'] = disparate_impact_favorable
        metrics_dict['unpriv_ratio_favorable_08'] = unpriv_ratio_favorable
        metrics_dict['priv_ratio_favorable_08'] = priv_ratio_favorable
        metrics_dict['disparate_impact_unfavorable_08'] = disparate_impact_unfavorable
        metrics_dict['unpriv_ratio_unfavorable_08'] = unpriv_ratio_unfavorable
        metrics_dict['priv_ratio_unfavorable_08'] = priv_ratio_unfavorable
        metrics_dict['priv_n_08'] = favorable_counts_df.iloc[0, 1]
        metrics_dict['unpriv_n_08'] = favorable_counts_df.iloc[1, 1]

    favorable_counts_df = results_df.groupby(by='is_aae_06').agg(
        {'pred': ['count', favorable, unfavorable]}).reset_index()

    if favorable_counts_df.shape == (2, 4):
        priv_total = favorable_counts_df.iloc[0, 1]
        unpriv_total = favorable_counts_df.iloc[1, 1]
        # favorable is outcome is not hate/harassment/abusive
        unpriv_ratio_favorable = safe_division(favorable_counts_df.iloc[1, 2], unpriv_total)
        priv_ratio_favorable = safe_division(favorable_counts_df.iloc[0, 2], priv_total)
        disparate_impact_favorable = safe_division(unpriv_ratio_favorable, priv_ratio_favorable)

        unpriv_ratio_unfavorable = safe_division(favorable_counts_df.iloc[1, 3], unpriv_total)
        priv_ratio_unfavorable = safe_division(favorable_counts_df.iloc[0, 3], priv_total)
        disparate_impact_unfavorable = safe_division(unpriv_ratio_unfavorable, priv_ratio_unfavorable)

        fpr_unpriv = safe_division(
            results_df[(results_df['is_aae_06'] == 1) & (results_df['pred'] == 1) & (results_df['label'] == 0)].shape[
                0],
            results_df[(results_df['is_aae_06'] == 1) & (results_df['label'] == 0)].shape[0]
        )
        fpr_priv = safe_division(
            results_df[(results_df['is_aae_06'] == 0) & (results_df['pred'] == 1) & (results_df['label'] == 0)].shape[
                0],
            results_df[(results_df['is_aae_06'] == 0) & (results_df['label'] == 0)].shape[0]
        )
        fpr_total = safe_division(
            results_df[(results_df['pred'] == 1) & (results_df['label'] == 0)].shape[0],
            results_df[(results_df['label'] == 0)].shape[0]
        )

        fnr_unpriv = safe_division(
            results_df[(results_df['is_aae_06'] == 1) & (results_df['pred'] == 0) & (results_df['label'] == 1)].shape[
                0],
            results_df[(results_df['is_aae_06'] == 1) & (results_df['label'] == 1)].shape[0]
        )
        fnr_priv = safe_division(
            results_df[(results_df['is_aae_06'] == 0) & (results_df['pred'] == 0) & (results_df['label'] == 1)].shape[
                0],
            results_df[(results_df['is_aae_06'] == 0) & (results_df['label'] == 1)].shape[0]
        )
        fnr_total = safe_division(
            results_df[(results_df['pred'] == 0) & (results_df['label'] == 1)].shape[0],
            results_df[(results_df['label'] == 1)].shape[0]
        )

        metrics_dict['unpriv_total_06'] = unpriv_total
        metrics_dict['priv_total_06'] = priv_total
        metrics_dict['fpr_unpriv_06'] = fpr_unpriv
        metrics_dict['fpr_priv_06'] = fpr_priv
        metrics_dict['fpr_total_06'] = fpr_total
        metrics_dict['fnr_unpriv_06'] = fnr_unpriv
        metrics_dict['fnr_priv_06'] = fnr_priv
        metrics_dict['fnr_total_06'] = fnr_total
        metrics_dict['disparate_impact_favorable_06'] = disparate_impact_favorable
        metrics_dict['unpriv_ratio_favorable_06'] = unpriv_ratio_favorable
        metrics_dict['priv_ratio_favorable_06'] = priv_ratio_favorable
        metrics_dict['disparate_impact_unfavorable_06'] = disparate_impact_unfavorable
        metrics_dict['unpriv_ratio_unfavorable_06'] = unpriv_ratio_unfavorable
        metrics_dict['priv_ratio_unfavorable_06'] = priv_ratio_unfavorable
        metrics_dict['priv_n_06'] = favorable_counts_df.iloc[0, 1]
        metrics_dict['unpriv_n_06'] = favorable_counts_df.iloc[1, 1]

    return metrics_dict


def strip_punc_hp(s):
    return str(s).translate(str.maketrans('', '', string.punctuation))


def remove_punctuation_tweet(text_array):
    # get rid of punctuation (except periods!)
    punctuation_no_period = "[" + re.sub("", "", string.punctuation) + "]"
    return np.array([re.sub(punctuation_no_period, "", text) for text in text_array])


def tfidf_vectorize(train_texts: np.ndarray,
                    train_labels: np.ndarray,
                    val_texts: np.ndarray,
                    test_texts: np.ndarray,
                    ngram_range: tuple = (1, 2),
                    top_k: int = 20000,
                    token_mode: str = 'word',
                    min_document_frequency: int = 2,
                    tf_idf: bool = True) -> tuple:
    """
    Vectorizes texts as n-gram vectors.

    1 text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams.

    # Arguments
        @:param train_texts: list, training text strings.
        @:param train_labels: np.ndarray, training labels.
        @:param val_texts: list, validation text strings.
        @:param ngram_range Range: (inclusive) of n-gram sizes for tokenizing text.
        @:param top_k: Limit on the number of features. We use the top 20K features.
        @:param token_mode:  Whether text should be split into word or character n-grams. One of 'word', 'char'.
        @:param min_document_frequency: Minimum document/corpus frequency below which a token will be discarded.

    # Returns
        x_train, x_val: vectorized training and validation texts

    # adapted from: https://developers.google.com/machine-learning/guides/text-classification/step-3
    """
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
        'ngram_range': ngram_range,
        'dtype': 'int32',
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'analyzer': token_mode,
        'min_df': min_document_frequency,
    }

    vectorizer = TfidfVectorizer(**kwargs) if tf_idf else CountVectorizer(**kwargs)
    train_texts = remove_punctuation_tweet(train_texts)
    val_texts = remove_punctuation_tweet(val_texts)
    test_texts = remove_punctuation_tweet(test_texts)
    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts)

    # Vectorize validation and test texts.
    x_val = vectorizer.transform(val_texts)
    x_test = vectorizer.transform(test_texts)

    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(top_k, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')
    x_test = selector.transform(x_test).astype('float32')

    return x_train, x_val, x_test


def glove_vectorize(train_texts,
                    val_texts,
                    test_texts,
                    path_to_glove_file):
    """
    Useful documentation:
    - https://keras.io/examples/nlp/pretrained_word_embeddings/
    - https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
    :param train_texts: nd.array of
    :param val_texts:
    :param test_texts:
    :return:
    """
    vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=200,
                                   standardize='lower_and_strip_punctuation')
    train_text_ds = tf.data.Dataset.from_tensor_slices(train_texts)

    vectorizer.adapt(train_text_ds)
    # tf.compat.v1.enable_eager_execution()

    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))

    embeddings_index = {}
    with open(path_to_glove_file, encoding="utf8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    print("Found %s word vectors." % len(embeddings_index))

    num_tokens = len(voc) + 2
    embedding_dim = 100
    hits = 0
    misses = 0

    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print(f"Converted {hits} words ({misses} misses)")

    x_train = vectorizer(train_texts).numpy()
    x_val = vectorizer(val_texts).numpy()
    x_test = vectorizer(test_texts).numpy()

    ## keep trainable=False so embeddings arent updated during training
    embedding_layer = Embedding(
        input_length=200,
        input_dim=num_tokens,
        output_dim=embedding_dim,
        embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
        trainable=False,
        name="glove_embeddings"
    )

    return x_train, x_val, x_test, embedding_layer


# def bert_tokenize(texts_ndarray, max_seq_length):
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # todo explore different bert models?
#     features = []
#     for example in texts_ndarray:
#         tokens = tokenizer.tokenize(example)
#         if len(tokens) > max_seq_length - 2:
#             tokens = tokens[:(max_seq_length - 2)]
#         tokens = ["[CLS]"] + tokens + ["[SEP]"]
#         input_ids = tokenizer.convert_tokens_to_ids(tokens)
#         length = len(input_ids)
#         padding = [0] * (max_seq_length - length)
#         input_ids += padding
#
#         features.append(input_ids)
#
#     return np.array(features)


# def prepare_bert_ds(train_data,
#                     validation_data,
#                     test_data,
#                     text_feature,
#                     label,
#                     max_seq_length):
#     train_encoded = bert_tokenize(train_data[text_feature].values, max_seq_length=max_seq_length)
#     dev_encoded = bert_tokenize(validation_data[text_feature].values, max_seq_length=max_seq_length)
#     test_encoded = bert_tokenize(test_data[text_feature].values, max_seq_length=max_seq_length)
#
#     return (train_encoded, train_data[label].values), (dev_encoded, validation_data[label].values), (
#         test_encoded, test_data[label].values)


def prepare_mindiff_ds(train_data,
                       validation_data,
                       test_data,
                       unpriv_label,
                       text_feature,
                       label,
                       batch_size,
                       max_seq_length):
    train_ds = tf.data.Dataset.from_tensor_slices((train_data[text_feature].values,
                                                   train_data[label].values.reshape(-1, 1) * 1.0)
                                                  ).batch(batch_size)

    dev_ds = tf.data.Dataset.from_tensor_slices((validation_data[text_feature].values,
                                                 validation_data[label].values.reshape(-1, 1) * 1.0)
                                                ).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((test_data[text_feature].values,
                                                  test_data[label].values.reshape(-1, 1) * 1.0)
                                                 ).batch(batch_size)

    unpriv_mask = train_data[unpriv_label] == 1
    priv_mask = train_data[unpriv_label] == 0

    true_negative_mask = train_data[label] == 0

    train_data_main = copy.copy(train_data)
    train_data_unpriv = train_data[true_negative_mask & unpriv_mask]
    train_data_priv = train_data[true_negative_mask & priv_mask]

    print(f'unpriv true negative count in {train_data_unpriv.shape[0]},' +
          f' priv true negative count: {train_data_priv.shape[0]}')

    train_texts_main, y_train_main = train_data_main[text_feature], train_data_main[label].values
    train_texts_unpriv, y_train_unpriv = train_data_unpriv[text_feature].values, train_data_unpriv[label].values
    train_texts_priv, y_train_priv = train_data_priv[text_feature].values, train_data_priv[label].values

    # convert pd.Dataframe to tf.Datasets
    dataset_train_main = tf.data.Dataset.from_tensor_slices((train_texts_main,
                                                             y_train_main.reshape(-1, 1) * 1.0)
                                                            ).batch(batch_size)
    dataset_train_unpriv = tf.data.Dataset.from_tensor_slices((train_texts_unpriv,
                                                               y_train_unpriv.reshape(-1, 1) * 1.0)
                                                              ).batch(batch_size)
    dataset_train_priv = tf.data.Dataset.from_tensor_slices((train_texts_priv,
                                                             y_train_priv.reshape(-1, 1) * 1.0)
                                                            ).batch(batch_size)
    # dataset_dev = tf.data.Dataset.from_tensor_slices((dev_texts,
    #                                                   y_dev.reshape(-1, 1) * 1.0)
    #                                                  ).batch(batch_size)
    # dataset_test = tf.data.Dataset.from_tensor_slices((test_texts,
    #                                                    y_test.reshape(-1, 1) * 1.0)
    #                                                   ).batch(batch_size)

    return dataset_train_main, dataset_train_unpriv, dataset_train_priv,\
           (validation_data[text_feature].values, validation_data[label].values),\
           (test_data[text_feature].values, test_data[label].values)


# def prepare_mindiff_ds_transformers(train_data,
#                                     validation_data,
#                                     test_data,
#                                     unpriv_label,
#                                     text_feature,
#                                     label,
#                                     batch_size,
#                                     max_seq_length):
#     (train_examples, y_train), (dev_examples, y_dev), (test_examples, y_test) = prepare_bert_ds(
#         train_data=train_data,
#         validation_data=validation_data,
#         test_data=test_data,
#         text_feature=text_feature,
#         label=label,
#         max_seq_length=max_seq_length,
#     )
#
#     unpriv_mask = train_data[unpriv_label] == 1
#     priv_mask = train_data[unpriv_label] == 0
#
#     true_negative_mask = train_data[label] == 0
#     unpriv_encoding_mask = np.array(true_negative_mask & unpriv_mask)
#     priv_encoding_mask = np.array(true_negative_mask & priv_mask)
#     unpriv_encoding_train = train_data[unpriv_encoding_mask][text_feature]
#     priv_encoding_train = train_data[priv_encoding_mask][text_feature]
#
#     print(f'unpriv true negative count in {np.shape(unpriv_encoding_train)},' +
#           f' priv true negative count: {np.shape(priv_encoding_train)}')
#
#     train_data_main = copy.copy(train_data)
#     train_data_unpriv = train_data[true_negative_mask & unpriv_mask]
#     train_data_priv = train_data[true_negative_mask & priv_mask]
#
#     train_texts_main, y_train_main = train_examples, train_data_main[label].values
#     train_texts_unpriv, y_train_unpriv = unpriv_encoding_train, train_data_unpriv[label].values
#     train_texts_priv, y_train_priv = priv_encoding_train, train_data_priv[label].values
#
#     # convert pd.Dataframe to tf.Datasets
#     dataset_train_main = tf.data.Dataset.from_tensor_slices((train_texts_main,
#                                                              y_train_main.reshape(-1, 1) * 1.0)
#                                                             ).batch(batch_size)
#     dataset_train_unpriv = tf.data.Dataset.from_tensor_slices((train_texts_unpriv,
#                                                                y_train_unpriv.reshape(-1, 1) * 1.0)
#                                                               ).batch(batch_size)
#     dataset_train_priv = tf.data.Dataset.from_tensor_slices((train_texts_priv,
#                                                              y_train_priv.reshape(-1, 1) * 1.0)
#                                                             ).batch(batch_size)
#     # dataset_dev = tf.data.Dataset.from_tensor_slices((dev_texts,
#     #                                                   y_dev.reshape(-1, 1) * 1.0)
#     #                                                  ).batch(batch_size)
#     # dataset_test = tf.data.Dataset.from_tensor_slices((test_texts,
#     #                                                    y_test.reshape(-1, 1) * 1.0)
#     #                                                   ).batch(batch_size)
#
#     return dataset_train_main, dataset_train_unpriv, dataset_train_priv, (dev_examples, y_dev), (test_examples, y_test)
#
#
# def prepare_mindiff_ds_transformers(train_data,
#                                     validation_data,
#                                     test_data,
#                                     unpriv_label,
#                                     text_feature,
#                                     label,
#                                     batch_size,
#                                     max_seq_length):
#     (train_examples, y_train), (dev_examples, y_dev), (test_examples, y_test) = prepare_bert_ds(
#         train_data=train_data,
#         validation_data=validation_data,
#         test_data=test_data,
#         text_feature=text_feature,
#         label=label,
#         max_seq_length=max_seq_length,
#     )
#
#     unpriv_mask = train_data[unpriv_label] == 1
#     priv_mask = train_data[unpriv_label] == 0
#
#     true_negative_mask = train_data[label] == 0
#     unpriv_encoding_mask = np.array(true_negative_mask & unpriv_mask)
#     priv_encoding_mask = np.array(true_negative_mask & priv_mask)
#     unpriv_encoding_train = train_examples[unpriv_encoding_mask, :]
#     priv_encoding_train = train_examples[priv_encoding_mask, :]
#
#     print(f'unpriv true negative count in {np.shape(unpriv_encoding_train)},' +
#           f' priv true negative count: {np.shape(priv_encoding_train)}')
#
#     train_data_main = copy.copy(train_data)
#     train_data_unpriv = train_data[true_negative_mask & unpriv_mask]
#     train_data_priv = train_data[true_negative_mask & priv_mask]
#
#     train_texts_main, y_train_main = train_examples, train_data_main[label].values
#     train_texts_unpriv, y_train_unpriv = unpriv_encoding_train, train_data_unpriv[label].values
#     train_texts_priv, y_train_priv = priv_encoding_train, train_data_priv[label].values
#
#     # convert pd.Dataframe to tf.Datasets
#     dataset_train_main = tf.data.Dataset.from_tensor_slices((train_texts_main,
#                                                              y_train_main.reshape(-1, 1) * 1.0)
#                                                             ).batch(batch_size)
#     dataset_train_unpriv = tf.data.Dataset.from_tensor_slices((train_texts_unpriv,
#                                                                y_train_unpriv.reshape(-1, 1) * 1.0)
#                                                               ).batch(batch_size)
#     dataset_train_priv = tf.data.Dataset.from_tensor_slices((train_texts_priv,
#                                                              y_train_priv.reshape(-1, 1) * 1.0)
#                                                             ).batch(batch_size)
#     # dataset_dev = tf.data.Dataset.from_tensor_slices((dev_texts,
#     #                                                   y_dev.reshape(-1, 1) * 1.0)
#     #                                                  ).batch(batch_size)
#     # dataset_test = tf.data.Dataset.from_tensor_slices((test_texts,
#     #                                                    y_test.reshape(-1, 1) * 1.0)
#     #                                                   ).batch(batch_size)
#
#     return dataset_train_main, dataset_train_unpriv, dataset_train_priv, (dev_examples, y_dev), (test_examples, y_test)
#

def logistic_regression_model(input_dim, embedding_layer=None, reg_strength=0):
    if embedding_layer is not None:
        return models.Sequential([
            embedding_layer,
            Flatten(),
            Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(l=reg_strength), name='logreg')  # output dim = 100
        ])
    else:
        return models.Sequential([
            tf.keras.layers.Dense(1,
                                  input_shape=(input_dim,),
                                  kernel_regularizer=tf.keras.regularizers.l2(l=reg_strength),
                                  activation='sigmoid')
        ])


def train_plot(history, output_dir, task_name, acc="acc"):
    plt.plot(history["loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.plot(history[f"{acc}"], label="train_acc")
    plt.plot(history[f"val_{acc}"], label="val_acc")
    plt.legend()
    plt.savefig(f'{output_dir}/{task_name}_training_plot.png')

def plot_cm(output_dir, task_name, labels, predictions, train_dev_test):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"Confusion Matrix: {train_dev_test}")
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{output_dir}/{task_name}_{train_dev_test}_cmplot.png')

def safe_division(x, y):
    if y == 0: return 0
    return x / y


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


# Plot yearly distributions of counts
def plot_yearly_distributions(pd_df, title, dataset_name, label):
    years = sorted(list(pd_df['DEP_YEAR'].unique()))
    num_years = len(years)
    dv_vals = sorted(list(pd_df[label].unique()))
    nrows, ncols = num_years, len(dv_vals)
    print(nrows, ncols)
    with plt.style.context('Solarize_Light2'):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * ncols, 10 * nrows))
        axes = axes.flatten()

        for dv, col, ax in zip(dv_vals * nrows, [col for col in years for _ in range(ncols)], axes):
            temp_df = pd_df[(pd_df[label] == dv) & (pd_df['DEP_YEAR'] == col)]
            temp_df['DEP_MONTH'].hist(by=temp_df[label], xrot=45, ax=ax, bins=12)
            delayed_title = "Delayed " if dv == 1 else "Not Delayed "
            ax.set_title(delayed_title + str(col) + ' Histogram')
            ax.set(xlabel=str(str(col) + ' Distribution'), ylabel=f'count of {str(col)}')

        plt.suptitle(title + ' Distributions', fontsize=25, verticalalignment='baseline')
        # plt.subplots_adjust(left=0.5)
        plt.tight_layout()
        plt.savefig(f"../reports/figures/{dataset_name}_year_histogram_matrix.png")
        plt.show()


# Plot distributions of counts
def plot_distributions(pd_df, title, dataset_name, label, cols_to_get_distr=None, compare_labels=False):
    if cols_to_get_distr == None:
        cols_to_get_distr = pd_df.columns.values
    num_cols = len(cols_to_get_distr)
    if compare_labels:
        dv_vals = list(pd_df[label].unique())
        dv_vals.sort()
        nrows, ncols = num_cols, len(dv_vals)
        print(nrows, ncols)
    else:
        nrows, ncols = get_middle_factors(num_cols)
    with plt.style.context('Solarize_Light2'):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10 * ncols, 10 * nrows))
        axes = axes.flatten()

        if compare_labels:
            for dv, col, ax in zip(dv_vals * nrows, [col for col in cols_to_get_distr for _ in range(ncols)], axes):
                temp_df = pd_df[pd_df[label] == dv]
                temp_df[col].hist(by=temp_df[label], xrot=45, ax=ax)
                delayed_title = "Delayed " if dv == 1 else "Not Delayed "
                ax.set_title(delayed_title + col + ' Histogram')
                ax.set(xlabel=str(col + ' Distribution'), ylabel=f'count of {col}')
        else:
            for col, ax in zip(cols_to_get_distr, axes):
                ax.hist(pd_df[col], histtype='bar')

                ax.set_title(col + ' Histogram')
                ax.set(xlabel=str(col + ' Distribution'), ylabel='count of {0}'.format(col))

        plt.suptitle(title + ' Distributions', fontsize=25, verticalalignment='baseline')
        # plt.subplots_adjust(left=0.5)
        plt.tight_layout()
        plt.savefig(f"../reports/figures/{dataset_name}_histogram_matrix.png")
        plt.show()


def plot_distribution(pd_df, title, col_to_get_distr):
    with plt.style.context('Solarize_Light2'):
        plt.figure(figsize=(18, 15))
        plt.hist(pd_df[col_to_get_distr], histtype='bar')
        plt.title(col_to_get_distr + ' distribution')
        plt.xlabel(col_to_get_distr)
        plt.ylabel('count')

        plt.show()


def compare_counts_boxplots(positive_pd, negative_pd, title, dataset_name, positive_label, cols=None):
    # unique_col_values = df_pd[feature_column].unique()
    # print unique_col_values
    if cols is None:
        cols = positive_pd.columns.values.tolist()

    data = []
    labels = []
    for col in cols:
        data.append(positive_pd[col])
        data.append(negative_pd[col])
        labels.append(f'{positive_label}_{col}')
        labels.append(f'not_{positive_label}_{col}')

    with plt.style.context('Solarize_Light2'):
        plt.figure(figsize=(35, 20))
        plt.ylabel('Counts')
        plt.title(title)
        plt.boxplot(data, showfliers=False)
        plt.xticks(np.arange(start=1, stop=len(data) + 1), labels)
        plt.savefig(f"../reports/figures/{dataset_name}_barplots.png")


def create_scatterplot_matrix(pd_df, dataset_name, cols_to_plot=None, label_column='temp'):
    sns.set(style="ticks")
    if cols_to_plot is None:
        cols_to_plot = pd_df.columns.values
        cols_to_plot = np.delete(cols_to_plot, np.where(cols_to_plot == label_column)).tolist()
    print(cols_to_plot)
    pairplot = sns.pairplot(pd_df, hue=label_column, markers=["o", "+"], vars=cols_to_plot)
    plt.savefig(f"../reports/figures/{dataset_name}_scatterplot_matrix.png")
    return pairplot


def plot_violin_distributions(pd_df, title, dataset_name, label_column, cols_to_plot=None):
    if cols_to_plot is None:
        cols_to_plot = pd_df.columns.values
        cols_to_plot = np.delete(cols_to_plot, np.where(cols_to_plot == label_column)).tolist()
        print(cols_to_plot)
    num_cols = len(cols_to_plot)
    nrows, ncols = get_middle_factors(num_cols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 22))
    axes = axes.flatten()

    for col, ax in zip(cols_to_plot, axes):
        ax.set_title(col + ' violinplot')
        create_violin_plot(pd_df, col, label_column, ax)
        ax.set(ylabel='count of {0}'.format(col))

    plt.suptitle(title, fontsize=16, verticalalignment='baseline')
    # plt.subplots_adjust(left=0.5)

    plt.savefig(f"../reports/figures/{dataset_name}_violinplot_matrix.png")


def create_violin_plot(pd_df, column, label_column, axes):
    sns.set(style="ticks")
    sns.violinplot(data=pd_df, x=label_column, y=column, ax=axes, kind='violin', cut=0)


def plot_distributions_2(pd_df, title, cols_to_get_distr, label):
    dv_vals = list(pd_df[label].unique())
    dv_vals.sort()
    num_axis = (len(cols_to_get_distr), len(dv_vals))
    with plt.style.context('Solarize_Light2'):
        fig, axes = plt.subplots(nrows=num_axis[0], ncols=num_axis[1], figsize=(5 * num_axis[1], 5 * num_axis[0]))
        axes = axes.flatten()
        for dv, col, ax in zip(dv_vals * num_axis[0], [col for col in cols_to_get_distr for _ in range(num_axis[1])],
                               axes):
            temp_df = pd_df[pd_df[label] == dv]
            temp_df[col].hist(by=temp_df[label], xrot=45, ax=ax)

            ax.set_title("Decile Score: " + str(dv) + " (" + col + ")")
            ax.set(xlabel=str(col + ' Distribution'), ylabel=f'count of {col}')

        plt.suptitle(title + ' Distributions', fontsize=30, verticalalignment='top')
        # plt.subplots_adjust(left=0.5)
        plt.savefig(f"./figs/{label}_histogram_matrix.png")
        plt.show()


def get_middle_factors(n: int) -> (int, int):
    step = 2 if n % 2 else 1
    factors = set(reduce(list.__add__, ([i, n // i] for i in range(1, int(np.sqrt(n)) + 1, step) if n % i == 0)))
    factors = np.sort(list(factors))
    print(factors)
    if (len(factors) > 3) & (len(factors) % 2 == 0):
        mid = int(len(factors) / 2)
        return factors[mid - 1], factors[mid]
    elif (len(factors) > 2) & (len(factors) % 2 != 0):
        mid = int(len(factors) / 2)
        return factors[mid], factors[mid]
    elif len(factors) > 1:
        return factors[0], factors[1]
    else:
        return 0


def plot_learning_curve(estimator, title, X, y, algorithm, dataset_name, model_name, y_lim=None, cv=None,
                        n_jobs=None,
                        train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    Generate a simple plot of the test and training learning curve.
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5)) (changed to np.linspace(0.1, 1.0, 10))
    """

    plt.figure()
    plt.title(title)
    if y_lim is not None:
        plt.ylim(*y_lim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    if model_name is 'dt_pruning_1' or model_name is 'boosting_1':
        N = y.shape[0]
        train_sizes = [50, 100] + [int(N * x / 10) for x in range(1, 8)]
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

    plt.savefig(f'./figs/learning_curve_{algorithm}_{model_name}_{dataset_name}')
    return plt


def plot_iterative_learning_curve(clfObj, trgX, trgY, tstX, tstY, params, model_name=None, dataset_name=None):
    # also adopted from jontays code
    np.random.seed(42)
    if model_name is None or dataset_name is None:
        raise
    cv = GridSearchCV(clfObj, n_jobs=1, param_grid=params, refit=True, verbose=10, cv=5, scoring='accuracy')
    cv.fit(trgX, trgY)
    regTable = pd.DataFrame(cv.cv_results_)
    regTable.to_csv('./output/ITER_base_{}_{}.csv'.format(model_name, dataset_name), index=False)
    d = defaultdict(list)
    name = list(params.keys())[0]
    for value in list(params.values())[0]:
        d['param_{}'.format(name)].append(value)
        clfObj.set_params(**{name: value})
        clfObj.fit(trgX, trgY)
        pred = clfObj.predict(trgX)
        d['train acc'].append(accuracy_score(trgY, pred))
        clfObj.fit(trgX, trgY)
        pred = clfObj.predict(tstX)
        d['test acc'].append(accuracy_score(tstY, pred))
        print(value)
    d = pd.DataFrame(d)
    d.to_csv('./output/ITERtestSET_{}_{}.csv'.format(model_name, dataset_name), index=False)
    return d


# def make_timing_curve(X_train, y_train, X_test, y_test, clf, model_name, dataset_name, alg):
#     # 'adopted' from JonTay's code
#     timing_df = defaultdict(dict)
#     for fraction in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#         st = clock()
#         np.random.seed(42)
#         clf.fit(X_train, y_train)
#         timing_df['train'][fraction] = clock() - st
#         st = clock()
#         clf.predict(X_test)
#         timing_df['test'][fraction] = clock() - st
#         print(model_name, dataset_name, fraction)
#     timing_df = pd.DataFrame(timing_df)
#     timing_df.to_csv(f'./output/{model_name}_{dataset_name}_timing.csv')
#
#     title = alg + ' ' + dataset_name + ' Timing Curve for Training and Prediction'
#     plot_model_timing(title, alg, model_name, dataset_name,
#                       timing_df.index.values * 100,
#                       pd.DataFrame(timing_df['train'], index=timing_df.index.values),
#                       pd.DataFrame(timing_df['test'], index=timing_df.index.values))
#     return timing_df


def plot_model_timing(title, algorithm, model_name, dataset_name, data_sizes, fit_scores, predict_scores, ylim=None):
    """
    Generate a simple plot of the given model timing data

    Parameters
    ----------
    title : string
        Title for the chart.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    data_sizes : list, array
        The data sizes

    fit_scores : list, array
        The fit/train times

    predict_scores : list, array
        The predict times

    """
    with plt.style.context('seaborn'):
        plt.close()
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training Data Size (% of total)")
        plt.ylabel("Time (s)")
        fit_scores_mean = np.mean(fit_scores, axis=1)
        fit_scores_std = np.std(fit_scores, axis=1)
        predict_scores_mean = np.mean(predict_scores, axis=1)
        predict_scores_std = np.std(predict_scores, axis=1)

        plt.fill_between(data_sizes, fit_scores_mean - fit_scores_std,
                         fit_scores_mean + fit_scores_std, alpha=0.2)
        plt.fill_between(data_sizes, predict_scores_mean - predict_scores_std,
                         predict_scores_mean + predict_scores_std, alpha=0.2)
        plt.plot(data_sizes, predict_scores_mean, 'o-', linewidth=1, markersize=4,
                 label="Predict time")
        plt.plot(data_sizes, fit_scores_mean, 'o-', linewidth=1, markersize=4,
                 label="Fit time")

        plt.legend(loc="best")
        plt.savefig(f'./figs/timing_curve_{algorithm}_{model_name}_{dataset_name}')
        plt.show()


def _save_cv_results(self):
    # TODO fix this
    regTable = pd.DataFrame(self.dt_model.cv_results_)
    regTable.to_csv(f'./output/cross_validation_{self.model_name}_{self.dataset_name}.csv',
                    index=False)

    results = pd.DataFrame(self.dt_model.cv_results_)
    components_col = 'param___n_components'
    best_clfs = results.groupby(components_col).apply(
        lambda g: g.nlargest(1, 'mean_test_score'))

    ax = plt.figure()
    best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score',
                   legend=False, ax=ax)
    ax.set_ylabel('Classification accuracy (val)')
    ax.set_xlabel('n_components')
    plt.savefig(f'./figs/cross_validation_{self.model_name}_{self.dataset_name}')
    plt.show()


# def save_model(dataset_name, estimator, file_name):
#     file_name = dataset_name + '_' + file_name + '_' + str(datetime.datetime.now().date()) + '.pkl'
#     joblib.dump(estimator, f'./models/{file_name}')


def balanced_accuracy(truth, pred):
    wts = compute_sample_weight('balanced', truth)
    return accuracy_score(truth, pred, sample_weight=wts)
