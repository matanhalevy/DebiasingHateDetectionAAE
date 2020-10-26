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
import requests

from src.data_cleaning.common import DataPreparer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)

OUTPUT_DIRECTORY = '../data'


class DavidsonTwitterPreparer(DataPreparer):
    """
    Davidson:
    count = number of CrowdFlower users who coded each tweet (min is 3, sometimes more users coded a tweet when judgments were determined to be unreliable by CF).

    hate_speech = number of CF users who judged the tweet to be hate speech.
    offensive_language = number of CF users who judged the tweet to be offensive.
    neither = number of CF users who judged the tweet to be neither offensive nor non-offensive.
    class = class label for majority of CF users. 0 - hate speech 1 - offensive language 2 - neither
    """

    def __init__(self, path_to_raw, overwrite=False, verbose=False):
        super().__init__(path_to_raw, overwrite, verbose)
        self.path_to_save_cleaned = 'twitter_datasets/cleaned/davidson_cleaned.csv'

    def load_data(self):
        self.raw_data = pd.read_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/davidson/davidson_labelled.csv', index_col=0)
        if self.verbose:
            logger.info(f'Uncleaned Twitter Davidson Shape: {self.raw_data.shape}')

    def prepare_data(self, append_aae=True):
        if os.path.exists(f'{OUTPUT_DIRECTORY}/{self.path_to_save_cleaned}') and not self.overwrite:
            self.clean_data = pd.read_csv(f'{OUTPUT_DIRECTORY}/{self.path_to_save_cleaned}')
        else:
            twitter_df = self.raw_data
            twitter_df['cleaned_tweet'] = twitter_df.apply(lambda row: self.preprocess_twitter_davidson(row['tweet']),
                                                           axis=1)
            twitter_df['is_hate'] = np.where(twitter_df['class'] == 0, 1, 0)
            twitter_df['is_offensive'] = np.where(twitter_df['class'] == 1, 1, 0)
            if append_aae:
                twitter_df = self.append_is_aae(twitter_df, 'cleaned_tweet')

            self.clean_data = twitter_df
            self.save_cleaned_data()
            self.save_test_train_dev_splits()

    def save_test_train_dev_splits(self):
        twitter_train, twitter_valid, twitter_test = self._create_test_train_dev_splits(tr_size=0.8,
                                                                                        dev_test_split=0.5,
                                                                                        stratify_label='is_hate')

        twitter_train.to_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/davidson/train.csv', index=False)
        twitter_valid.to_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/davidson/dev.csv', index=False)
        twitter_test.to_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/davidson/test.csv', index=False)


class FountaTwitterPreparer(DataPreparer):
    '''
    Founta et al:
    80k
    - literature link: https://datalab.csd.auth.gr/wp-content/uploads/publications/17909-77948-1-PB.pdf
    - dataset Link: https://dataverse.mpi-sws.org/dataset.xhtml?persistentId=doi:10.5072/FK2/ZDTEMN
    'abusive', 'normal', 'hateful', 'spam' & nan as maj_label
    '''

    def __init__(self, path_to_raw, overwrite=False, verbose=False):
        super().__init__(path_to_raw, overwrite, verbose)
        self.path_to_save_cleaned = 'twitter_datasets/cleaned/founta_cleaned.csv'

    def load_data(self):
        self.raw_data = pd.read_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/founta/hatespeechtwitter.csv')
        if self.verbose:
            logger.info(f'Uncleaned Twitter Founta Shape: {self.raw_data.shape}')

    def prepare_data(self, append_aae=True):
        if os.path.exists(f'{OUTPUT_DIRECTORY}/{self.path_to_save_cleaned}') and not self.overwrite:
            self.clean_data = pd.read_csv(f'{OUTPUT_DIRECTORY}/{self.path_to_save_cleaned}')
        else:
            self.raw_data.dropna(inplace=True)  # 4 instances
            self.raw_data['is_hate'] = np.where(self.raw_data['maj_label'] == 'hateful', 1, 0)
            self.raw_data['is_abusive'] = np.where(self.raw_data['maj_label'] == 'abusive', 1, 0)

            if os.path.exists(f'{OUTPUT_DIRECTORY}/twitter_datasets/founta/with_tweets.csv'):
                founta_twitter = pd.read_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/founta/with_tweets.csv')
            else:
                founta_tweets_df = self.create_corresponding_tweets_df()
                # 79996 rows originally, only able to retrieve 50398 tweets
                founta_twitter = self.raw_data.merge(founta_tweets_df[['id', 'text']], left_on='tweet_id', right_on='id',
                                                     how='left')
                founta_twitter.dropna(inplace=True)
                founta_twitter.drop(columns=['id'])
                founta_twitter.to_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/founta/with_tweets.csv', index=False)

            founta_twitter['cleaned_tweet'] = founta_twitter.apply(
                lambda row: self.preprocess_twitter_davidson(row['text']), axis=1)
            if append_aae:
                founta_twitter = self.append_is_aae(founta_twitter, 'cleaned_tweet')

            self.clean_data = founta_twitter
            self.save_cleaned_data()
            self.save_test_train_dev_splits()

    def save_test_train_dev_splits(self):
        twitter_train, twitter_valid, twitter_test = self._create_test_train_dev_splits(tr_size=0.8,
                                                                                        dev_test_split=0.5,
                                                                                        stratify_label='is_hate')
        # train: 861 7879 0.1093%   test: 214 1962 0.1091    dev: 121 1103 0.1097%

        twitter_train.to_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/founta/train.csv', index=False)
        twitter_valid.to_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/founta/dev.csv', index=False)
        twitter_test.to_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/founta/test.csv', index=False)


class WaseemTwitterPreparer(DataPreparer):
    '''
    If using NAACL_SRW_2016.csv please cite using:

    @InProceedings{waseem-hovy:2016:N16-2,
      author    = {Waseem, Zeerak  and  Hovy, Dirk},
      title     = {Hateful Symbols or Hateful People? Predictive Features for Hate Speech Detection on Twitter},
      booktitle = {Proceedings of the NAACL Student Research Workshop},
      month     = {June},
      year      = {2016},
      address   = {San Diego, California},
      publisher = {Association for Computational Linguistics},
      pages     = {88--93},
      url       = {http://www.aclweb.org/anthology/N16-2013}
    }

    If using NLP+CSS_2016.csv please cite using:

    @InProceedings{waseem:2016:NLPandCSS,
      author    = {Waseem, Zeerak},
      title     = {Are You a Racist or Am I Seeing Things? Annotator Influence on Hate Speech Detection on Twitter},
      booktitle = {Proceedings of the First Workshop on NLP and Computational Social Science},
      month     = {November},
      year      = {2016},
      address   = {Austin, Texas},
      publisher = {Association for Computational Linguistics},
      pages     = {138--142},
      url       = {http://aclweb.org/anthology/W16-5618}
    }

    '''

    def __init__(self, path_to_raw, overwrite=False, verbose=False):
        super().__init__(path_to_raw, overwrite, verbose)
        self.path_to_save_cleaned = 'twitter_datasets/cleaned/waseem_cleaned.csv'

    def load_data(self):
        hovy_waseem = pd.read_csv(self.path_to_raw[0], header=None)
        hovy_waseem.columns = ['tweet_id', 'label']
        waseem = pd.read_csv(self.path_to_raw[1], delim_whitespace=True)
        # choose expert label todo verify what they did in paper
        waseem_expert = waseem[['TweetID', 'Expert']]
        waseem_expert.columns = ['tweet_id', 'label']

        hovy_waseem.loc[:,'original_ds'] = 'hovy_waseem_2016'
        waseem_expert.loc[:,'original_ds'] = 'waseem_2016'
        self.raw_data = pd.concat([hovy_waseem, waseem_expert])
        if self.verbose:
            logger.info(f'Uncleaned NAACL_SRW_2016 Waseem Hovy Shape: {hovy_waseem.shape}')
            logger.info(f'Uncleaned NLP+CSS 2016 Waseem Shape: {waseem_expert.shape}')
            logger.info(f'Merged uncleaned Waseem Shape: {self.raw_data.shape}')

    def prepare_data(self, append_aae=True):
        if os.path.exists(f'{OUTPUT_DIRECTORY}/{self.path_to_save_cleaned}') and not self.overwrite:
            self.clean_data = pd.read_csv(f'{OUTPUT_DIRECTORY}/{self.path_to_save_cleaned}')
        else:
            # label types = ['racism' 'sexism' 'none' 'neither' 'both']
            self.raw_data['is_racism'] = np.where(
                np.logical_or(self.raw_data['label'] == 'racism', self.raw_data['label'] == 'both'), 1, 0)
            self.raw_data['is_sexism'] = np.where(
                np.logical_or(self.raw_data['label'] == 'sexism', self.raw_data['label'] == 'both'), 1, 0)
            self.raw_data['is_hate'] = np.where(np.logical_or(np.logical_or(self.raw_data['label'] == 'racism',
                                                              self.raw_data['label'] == 'sexism'),
                                                              self.raw_data['label'] == 'both'), 1, 0)

            if os.path.exists(f'{OUTPUT_DIRECTORY}/twitter_datasets/waseem/with_tweets.csv'):
                waseem_twitter = pd.read_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/waseem/with_tweets.csv')
            else:
                waseem_tweets_df = self.create_corresponding_tweets_df()
                # 31762 rows originally, only able to retrieve 24648 tweets, there are duplicates
                waseem_twitter = self.raw_data.merge(waseem_tweets_df[['id', 'text']], left_on='tweet_id',
                                                     right_on='id', how='left')
                waseem_twitter.dropna(inplace=True)
                waseem_twitter.drop(columns=['id'])
                waseem_twitter.drop_duplicates(inplace=True) # 16631 after
                waseem_twitter.to_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/waseem/with_tweets.csv', index=False)

            waseem_twitter['cleaned_tweet'] = waseem_twitter.apply(
                lambda row: self.preprocess_twitter_davidson(row['text']), axis=1)
            if append_aae:
                waseem_twitter = self.append_is_aae(waseem_twitter, 'cleaned_tweet')

            if self.verbose:
                logger.info(f'Cleaned Waseem Twitter DS Shape: {waseem_twitter.shape}')
            self.clean_data = waseem_twitter
            self.save_cleaned_data()
            self.save_test_train_dev_splits()

    def save_test_train_dev_splits(self):
        twitter_train, twitter_valid, twitter_test = self._create_test_train_dev_splits(tr_size=0.8,
                                                                                        dev_test_split=0.5,
                                                                                        stratify_label='is_hate')
        # train: 861 7879 0.1093%   test: 214 1962 0.1091    dev: 121 1103 0.1097%

        twitter_train.to_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/waseem/train.csv', index=False)
        twitter_valid.to_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/waseem/dev.csv', index=False)
        twitter_test.to_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/waseem/test.csv', index=False)


class GolbeckTwitterPreparer(DataPreparer):
    """
    Golbeck given as tdf, converted to csv by opening in Excel.
    15.7% were labelled as harassment (5495-29505 negative examples)

    Labelled by two, if they disagreed brought in a third coder. (2711/35k required a third coder)
    All encompassing Harassment label

    TODO:  There is duplicate text tweets, may want to remove.
    """
    def __init__(self, path_to_raw, overwrite=False, verbose=False):
        super().__init__(path_to_raw, overwrite, verbose)
        self.path_to_save_cleaned = 'twitter_datasets/cleaned/golbeck_cleaned.csv'

    def load_data(self):
        self.raw_data = pd.read_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/golbeck/onlineHarassmentDataset.csv', index_col='ID')
        self.raw_data = self.raw_data[['Code', 'Tweet']]
        if self.verbose:
            logger.info(f'Uncleaned Twitter Golbeck Shape: {self.raw_data.shape}')

    def prepare_data(self, append_aae=True):
        if os.path.exists(f'{OUTPUT_DIRECTORY}/{self.path_to_save_cleaned}') and not self.overwrite:
            self.clean_data = pd.read_csv(f'{OUTPUT_DIRECTORY}/{self.path_to_save_cleaned}')
        else:
            golbeck = self.raw_data
            golbeck['is_hate'] = np.where(golbeck['Code'] == 'H', 1, 0)
            golbeck['cleaned_tweet'] = golbeck.apply(lambda row: self.preprocess_twitter_davidson(row['Tweet']),
                                                           axis=1)

            golbeck.drop_duplicates(subset=['cleaned_tweet'], inplace=True)
            if append_aae:
                golbeck = self.append_is_aae(golbeck, 'cleaned_tweet')

            self.clean_data = golbeck
            self.save_cleaned_data()
            self.save_test_train_dev_splits()

    def save_test_train_dev_splits(self):
        twitter_train, twitter_valid, twitter_test = self._create_test_train_dev_splits(tr_size=0.8,
                                                                                        dev_test_split=0.5,
                                                                                        stratify_label='is_hate')

        twitter_train.to_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/golbeck/train.csv', index=False)
        twitter_valid.to_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/golbeck/dev.csv', index=False)
        twitter_test.to_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/golbeck/test.csv', index=False)