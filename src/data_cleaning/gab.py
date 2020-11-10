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

from src.data_cleaning.common import DataPreparer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)

OUTPUT_DIRECTORY = '../data'


class GabPreparer(DataPreparer):
    """
    processing of this dataset is done from section 4 of the paper here: https://osf.io/nqt6h/
    CV ("Call for Violence"), HD (an "Assault on Human Dignity"), HD and CV, or NH ("Not Hateful").
    If none apply, the document is to be considered NH ("Not Hateful"). Vulgarity/Offensive Language directed at an indi-
    vidual (VO)

    Targeted Populations
    This column should be used to identify which types of groups are targeted by
    the hateful rhetoric. Select all descriptors which apply.
    - RAE: Race or ethnicity (includes anti-asian, anti-latino, anti-black, anti-arab, anti-semitism etc.)
    - NAT: Nationality/regionalism (includes general xenophobia and targets against countries/regions)
    - GEN: Gender (anti-woman, anti-man, anti-trans etc.)
    - REL: Religion/spiritual beliefs (anti-muslim, anti-christian, etc.)
    - SXO: Sexual Orientation
    - IDL: Ideology (conservative/liberal/leftist/right-wing)
    - POL: Political identification. Includes any reference to membership in a political organization (Democratic/Republican/ etc.).
    - MPH: Mental/physical health status, physical disability.

    Also EX/IM if the speech is explicit or implicit.

    - turned into binary label with union of the cv/hd
    """

    def __init__(self, path_to_raw, verbose=False):
        super().__init__(path_to_raw, verbose)
        self.path_to_save_cleaned = 'gab/gab_cleaned_with_aae.csv'

    def load_data(self):
        self.raw_data = pd.read_csv(f'{OUTPUT_DIRECTORY}/gab/GabHateCorpus_annotations.tsv', sep='\t', header=0)
        if self.verbose:
            unique_posts = len(self.raw_data["ID"].unique())
            logger.info(f'Uncleaned GAB Shape: {self.raw_data.shape}')  # (86529, 17)
            logger.info(f'# of Unique Posts in GAB: {unique_posts}')  # 27665

    def prepare_data(self, append_aae=True):
        # add binary label and investigation of aggregation methods
        counts_df = self.raw_data.groupby(["ID"]).agg(num_annotators=('ID', 'count'),
                                                      num_hate_label=('Hate', 'sum'),
                                                      median_hate_label=('Hate', 'median'),
                                                      mean_hate_label=('Hate', 'mean'))

        if self.verbose:
            counts_df['round_mean_hate_label'] = counts_df['mean_hate_label']
            counts_df = counts_df.round({'round_mean_hate_label': 0})

            logger.info('num of hate labels using median:',
                        len(counts_df.loc[counts_df['median_hate_label'] == 1]))  # 2337, in paper there is 2399
            logger.info('num of hate labels using mean rounded >0.5:',
                        len(counts_df.loc[counts_df['round_mean_hate_label'] == 1]))  # 2337, in paper there is 2399
            logger.info('num of hate labels using >= 1:', len(counts_df.loc[counts_df['num_hate_label'] >= 1]))  # 7813
            logger.info('num of hate labels using >1:', len(counts_df.loc[counts_df['num_hate_label'] > 1]))  # 2607
            logger.info('num_annotators summary stats:', counts_df['num_annotators'].describe())

        gab_data = self.raw_data.join(how='left', other=counts_df[['median_hate_label']], on='ID')
        gab_data.fillna(0, inplace=True)

        # assuming here median label is majority
        cleaned_gab_data = gab_data.groupby('ID', ).agg(
            Text=('Text', 'first'),
            text_id=('ID', 'first'),
            is_hate=('median_hate_label', 'median'),
            hd=('HD', 'median'),
            cv=('CV', 'median'),
            vo=('VO', 'median'),
            rel=('REL', 'median'),
            rae=('RAE', 'median'),
            sxo=('SXO', 'median'),
            gen=('GEN', 'median'),
            idl=('IDL', 'median'),
            nat=('NAT', 'median'),
            pol=('POL', 'median'),
            mph=('MPH', 'median'),
            ex=('EX', 'median'),
            im=('IM', 'median'),
        )

        if append_aae:
            # In GAB only 1 post is marked as aae by dialect (this one is all east asian characters that look like emojis)
            cleaned_gab_data = self.append_is_aae(cleaned_gab_data, 'Text')

        cleaned_gab_data = self._append_target_pop(cleaned_gab_data)
        self.clean_data = cleaned_gab_data
        self.save_cleaned_data()
        self.save_test_train_dev_splits()

    def _append_target_pop(self, gab_data):
        hot_encoded_gab = gab_data[['rae', 'rel', 'sxo', 'gen', 'idl', 'nat', 'pol', 'mph']]
        hot_encoded_gab['sum_labels'] = hot_encoded_gab.sum(axis=1)
        if self.verbose:
            pop_target_gr_1 = hot_encoded_gab[hot_encoded_gab['sum_labels'] > 1].shape[0]
            no_pop_target = hot_encoded_gab[hot_encoded_gab['sum_labels'] == 0].shape[0]
            logger.info(
                f'There are {pop_target_gr_1} instances where there is more than one labelled targeted population.')  # 244 / 27665 rows, 1911 rows > 0
            logger.info(
                f'There are {no_pop_target} instances without a labelled targeted population.')  # 25754 / 27665 rows
        hot_encoded_gab['no_tp'] = np.where(hot_encoded_gab['sum_labels'] == 0, 1, 0)
        hot_encoded_gab.drop(columns=['sum_labels'], inplace=True)
        target_pop_label = pd.Series(hot_encoded_gab.columns[np.where(hot_encoded_gab != 0)[1]])
        gab_data['target_pop'] = target_pop_label
        return gab_data

    def save_test_train_dev_splits(self):
        # gab stratified on target pop. 24353 train, 1716 valid, 1586 test, 88/0.062/0.058
        gab_train, gab_test, gab_valid = self._create_test_train_dev_splits(tr_size=0.88, dev_test_split=0.5,
                                                                            stratify_label='target_pop')
        # train: 2059/24345 0.084575%   test: 139/1660 0.084575% <- paper has more % labelled as hate (0.23)   dev: 139/1660 0.084575%

        gab_train.to_json(orient='records', lines=True, index=True,
                          path_or_buf=f'{OUTPUT_DIRECTORY}/gab/majority_gab_dataset_25k/train.jsonl')
        gab_test.to_json(orient='records', lines=True, index=True,
                         path_or_buf=f'{OUTPUT_DIRECTORY}/gab/majority_gab_dataset_25k/test.jsonl')
        gab_valid.to_json(orient='records', lines=True, index=True,
                          path_or_buf=f'{OUTPUT_DIRECTORY}/gab/majority_gab_dataset_25k/dev.jsonl')
