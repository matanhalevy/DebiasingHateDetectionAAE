import string

import pandas as pd
import numpy as np
import os
import gc
import subprocess

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, roc_auc_score
from transformers import DataCollatorForLanguageModeling, LineByLineTextDataset
import matplotlib.pyplot as plt
import seaborn as sns

# todo pass as param
root = 'drive/MyDrive/HateSpeech/benchmarking'


def get_line_by_line_ds(txt_path, tokenizer, max_seq_length):
    return LineByLineTextDataset(
        file_path=txt_path,
        tokenizer=tokenizer,
        block_size=max_seq_length,
    )


def get_data_collator_for_lm(tokenizer, mlm_prob=0.15):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_prob,
    )


def create_test_train_dev_splits(clean_data, tr_size=0.8, dev_test_split=0.5, stratify_label='is_harassment', seed=42):
    train, valid_test = train_test_split(clean_data, train_size=tr_size, random_state=seed,
                                         stratify=clean_data[[stratify_label]])
    test, valid = train_test_split(valid_test, train_size=dev_test_split, random_state=seed,
                                   stratify=valid_test[stratify_label])

    print(f"Train - # of f{stratify_label}: {train[train[stratify_label] == 1].shape[0]}, total instances: " +
          f"{train.shape[0]}, percent of positive: {train[train[stratify_label] == 1].shape[0] / train.shape[0]}")
    print(f"Test - # of f{stratify_label}: {test[test[stratify_label] == 1].shape[0]}, total instances: " +
          f"{test.shape[0]}, percent of positive: {test[test[stratify_label] == 1].shape[0] / test.shape[0]}")
    print(f"Test - # of f{stratify_label}: {valid[valid[stratify_label] == 1].shape[0]}, total instances: " +
          f"{valid.shape[0]}, percent of positive: {valid[valid[stratify_label] == 1].shape[0] / valid.shape[0]}")

    return train, test, valid


def save_test_train_dev_splits(ds, subfolder='combined_harassment', stratify_label='is_harassment'):
    save_folder = f'{root}/data/twitter/{subfolder}'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    twitter_train, twitter_valid, twitter_test = create_test_train_dev_splits(ds,
                                                                              tr_size=0.8,
                                                                              dev_test_split=0.5,
                                                                              stratify_label=stratify_label,
                                                                              )
    # train: 861 7879 0.1093%   test: 214 1962 0.1091    dev: 121 1103 0.1097%

    twitter_train.to_csv(f'{save_folder}/train.csv', index=False)
    twitter_valid.to_csv(f'{save_folder}/dev.csv', index=False)
    twitter_test.to_csv(f'{save_folder}/test.csv', index=False)

    return twitter_train, twitter_valid, twitter_test


def safe_division(x, y):
    if y == 0: return 0
    return x / y

def compute_metrics_trainer(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    preds_prob = torch.sigmoid(torch.tensor(pred.predictions)).numpy()[:,1]
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    fpr = safe_division(fp,(fp+tn))
    fnr = safe_division(fn,(fn+tp))
    try:
        auc_roc = roc_auc_score(y_true=labels, y_score=preds_prob)
    except ValueError:
        auc_roc = 0.
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc_roc': auc_roc,
        'fpr': fpr,
        'fnr': fnr,
    }


def compute_basic_metrics(labels, preds, preds_prob):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    fpr = safe_division(fp, (fp + tn))
    fnr = safe_division(fn, (fn + tp))
    try:
        auc_roc = roc_auc_score(y_true=labels, y_score=preds_prob)
    except ValueError:
        auc_roc = 0.
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc_roc': auc_roc,
        'fpr': fpr,
        'fnr': fnr,
    }

def compute_metrics_ensemble(labels, preds, preds_prob, in_group_labels):
    results = compute_basic_metrics(labels, preds, preds_prob)
    return compute_fairness_metrics(results, preds, in_group_labels, labels)

def compute_fairness_metrics(metrics_dict, preds, in_group_labels_06, true_labels):
    results_df = pd.DataFrame()
    results_df['pred'] = preds
    results_df['label'] = true_labels
    results_df['is_aae_06'] = in_group_labels_06

    def favorable(series):
        favorable_ser = series[series == 0]
        return len(favorable_ser)

    def unfavorable(series):
        unfavorable_ser = series[series == 1]
        return len(unfavorable_ser)

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

def plot_cm(output_dir, task_name, labels, predictions, train_dev_test):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"Confusion Matrix: {train_dev_test}")
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(f'{output_dir}/{task_name}_{train_dev_test}_cmplot.png')

def strip_punc_hp(s):
    return str(s).translate(str.maketrans('', '', string.punctuation))

def show_gpu(msg):
    """
    ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """

    def query(field):
        return (subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={field}',
             '--format=csv,nounits,noheader'],
            encoding='utf-8'))

    def to_int(result):
        return int(result.strip().split('\n')[0])

    used = to_int(query('memory.used'))
    total = to_int(query('memory.total'))
    pct = used / total
    print('\n' + msg, f'{100 * pct:2.1f}% ({used} out of {total})')


# clean texts
def preprocess_twitter_davidson(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned

    # TODO this is taken from Davidson 2017, consider removing !!!!!! on retweets and extra :
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    # parsed_text = parsed_text.code("utf-8", errors='ignore')
    return parsed_text
