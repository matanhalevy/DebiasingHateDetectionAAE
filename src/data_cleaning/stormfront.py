import logging
import json
import re
import logging
import os
import sys
import tarfile
import time
from pathlib import Path
import pandas as pd
import numpy as np

from src.data_cleaning import DataPreparer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)

OUTPUT_DIRECTORY = '../data'


class StormfrontPreparer(DataPreparer):
    """""
    More information about the dataset and the guidelines can be found in the following article https://www.aclweb.org/anthology/W18-51.pdf
    """""

    def __init__(self, path_to_raw, verbose=False):
        super().__init__(path_to_raw, verbose)
        self.path_to_save_cleaned = 'white_supremacy/cleaned_stormfront.csv'

    def load_data(self):
        self.raw_data = pd.read_csv(f'{OUTPUT_DIRECTORY}/stormfront/annotations_metadata.csv')
        if self.verbose:
            unique_posts = len(self.raw_data["ID"].unique())
            logger.info(f'Uncleaned Stormfront Shape: {self.raw_data.shape}')
            logger.info(f'# of Unique Posts in Stormfront: {unique_posts}')

    def prepare_data(self, append_aae=True):
        sf_content_dict = dict()
        for file in self.raw_data['file_id']:
            with open(f'{OUTPUT_DIRECTORY}/stormfront/all_files/{file}.txt', "r", encoding='utf8') as sf_file:
                sf_content_dict[file] = sf_file.read()

        sf_text_df = pd.DataFrame.from_dict(data=sf_content_dict, orient='index', columns=['text'])
        sf_text_df.index.names = ['file_id']
        stormfront = self.raw_data.join(how='left', other=sf_text_df, on='file_id')

        if append_aae:
            # In Stormfront no sentences are marked as aae
            stormfront = self.append_is_aae(stormfront, 'text')  # 0 marked as aae

        stormfront['is_hate'] = np.where(stormfront['label'] == 'hate', 1, 0)
        stormfront.drop(columns=['label'], inplace=True)
        self.clean_data = stormfront
        self.save_cleaned_data()
        self.save_test_train_dev_splits()

    def save_test_train_dev_splits(self):
        # gab stratified on target pop. 24353 train, 1716 valid, 1586 test, 88/0.062/0.058
        stormfront_train, stormfront_valid, stormfront_test = self._create_test_train_dev_splits(tr_size=0.88,
                                                                                                 dev_test_split=0.5,
                                                                                                 stratify_label='is_hate')
        # train: 861 7879 0.1093%   test: 214 1962 0.1091    dev: 121 1103 0.1097%

        stormfront_train.to_csv(f'{OUTPUT_DIRECTORY}/white_supremacy/train.tsv', sep='\t')
        stormfront_valid.to_csv(f'{OUTPUT_DIRECTORY}/white_supremacy/dev.tsv', sep='\t')
        stormfront_test.to_csv(f'{OUTPUT_DIRECTORY}/white_supremacy/test.tsv', sep='\t')
