import pandas as pd
import numpy as np
import os
from tokenizers.implementations import BertWordPieceTokenizer
from transformers import BertConfig, BertForMaskedLM, LineByLineTextDataset, BertTokenizerFast, BertTokenizer
from transformers import BertForSequenceClassification

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


def download_base_bert_tokenizer(bert_uncased_tokenizer_path):
    """
    If need to download the BertTokenizer base vocab file on the machine, call method
    :param bert_uncased_tokenizer_path: save path for tokenizer model
    """
    # Save existing BERT Tokenizer as baseline for AAE Tokenizer
    slow_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    if not os.path.exists(bert_uncased_tokenizer_path):
        os.makedirs(bert_uncased_tokenizer_path)
    slow_tokenizer.save_pretrained(bert_uncased_tokenizer_path)


def create_aae_tokenizer(bert_uncased_tokenizer_path,
                         blodgett_cleaned_aae_save_path,
                         aae_tokenizer_path,
                         ):
    # check that tokenizer from tokenizers library exist
    if not os.path.exists(f'{aae_tokenizer_path}/tokenizer.json'):
        download_base_bert_tokenizer(bert_uncased_tokenizer_path)

    # Load the fast tokenizer from saved bert file
    tokenizer = BertWordPieceTokenizer(
        f'{bert_uncased_tokenizer_path}bert-base-uncased-vocab.txt',
        lowercase=True,
        strip_accents=True,
    )

    tokenizer.train(
        blodgett_cleaned_aae_save_path,
        min_frequency=2,
    )

    if not os.path.exists(aae_tokenizer_path):
        os.makedirs(aae_tokenizer_path)

    tokenizer.save(f'{aae_tokenizer_path}/tokenizer.json')


def get_aae_tokenizer(aae_tokenizer_path,
                      bert_uncased_tokenizer_path,
                      blodgett_cleaned_aae_save_path,
                      max_seq_length,
                      ):
    # check that tokenizer from tokenizers library exist
    if not os.path.exists(f'{aae_tokenizer_path}/tokenizer.json'):
        create_aae_tokenizer(bert_uncased_tokenizer_path=bert_uncased_tokenizer_path,
                             blodgett_cleaned_aae_save_path=blodgett_cleaned_aae_save_path,
                             aae_tokenizer_path=aae_tokenizer_path
                             )

    # load AAE Tokenizer from tokenizers library
    return BertTokenizerFast.from_pretrained(aae_tokenizer_path, max_len=max_seq_length)
