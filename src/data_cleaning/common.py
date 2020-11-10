import json
import re
import logging
import os
import sys
import tarfile
import time
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import numpy as np
import xml.etree.ElementTree as et

import requests

from ..existing_models.twitteraae.code.predict import Predict
from sklearn.model_selection import train_test_split

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)

OUTPUT_DIRECTORY = '../data'


# TODO: use path_to_raw instead of hardcoding
# TODO: Don't prepare data if cleaned data already exists


# This assumes we don't need to drop columns that won't be used for modelling, remove prior to saving otherwise.
class DataPreparer:
    def __init__(self, path_to_raw, overwrite=False, verbose=False):
        self.path_to_raw = path_to_raw
        self.overwrite = overwrite
        self.verbose = verbose
        self.raw_data = None
        self.clean_data = None
        self.path_to_save_cleaned = None

        load_dotenv()

    def load_data(self):
        pass

    def get_raw_data(self):
        '''
        ensure data is loaded first
        :return: raw pandas dataset in
        '''
        return self.raw_data

    def get_cleaned_data(self):
        '''
        ensure cleaned data is prepared first
        :return: cleaned panda dataset
        '''
        return self.clean_data

    def save_cleaned_data(self):
        self.clean_data.to_csv(f'{OUTPUT_DIRECTORY}/{self.path_to_save_cleaned}', index=False)

    def append_is_aae(self, cleaned_df, text_label):
        # Output proportions are for the African-American, Hispanic, Asian, and White topics for text
        # recommended to assign dialict labels to tweets with dialect probabilities > 80%
        dialect_model = self._get_dialect_model()

        cleaned_df['dialect_prs'] = cleaned_df.apply(
            lambda row: dialect_model.predict(row[text_label].split()), axis=1)
        cleaned_df['is_aae_08'] = cleaned_df['dialect_prs'].str[0] >= 0.8
        cleaned_df['is_aae_08'] = cleaned_df['is_aae_08'].astype(int)
        cleaned_df['is_aae_06'] = cleaned_df['dialect_prs'].str[0] >= 0.6
        cleaned_df['is_aae_06'] = cleaned_df['is_aae_06'].astype(int)

        return cleaned_df

    def _create_test_train_dev_splits(self, tr_size=0.8, dev_test_split=0.5, stratify_label='is_hate'):
        train, valid_test = train_test_split(self.clean_data, train_size=tr_size, random_state=42,
                                             stratify=self.clean_data[[stratify_label]])
        test, valid = train_test_split(valid_test, train_size=dev_test_split, random_state=42,
                                       stratify=valid_test[stratify_label])

        if self.verbose:
            logger.info(f"Train - # of is_hate: {train[train['is_hate'] == 1].shape[0]}, total instances: " +
                        f"{train.shape[0]}, percent of positive: {train[train['is_hate'] == 1].shape[0] / train.shape[0]}")
            logger.info(f"Test - # of is_hate: {test[test['is_hate'] == 1].shape[0]}, total instances: " +
                        f"{test.shape[0]}, percent of positive: {test[test['is_hate'] == 1].shape[0] / test.shape[0]}")
            logger.info(f"Test - # of is_hate: {valid[valid['is_hate'] == 1].shape[0]}, total instances: " +
                        f"{valid.shape[0]}, percent of positive: {valid[valid['is_hate'] == 1].shape[0] / valid.shape[0]}")

        return train, test, valid

    def save_test_train_dev_splits(self):
        pass

    def preprocess_twitter_davidson(self, text_string):
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

    @staticmethod
    def get_tweets_call(tweet_ids: str) -> json:
        '''
        https://developer.twitter.com/en/docs/twitter-api/tweets/lookup/quick-start
        takes in up to 100 tweet ids and returns the content of the tweet (text), you can add more parameters but I didn't need

        :return:
        '''
        params = {'ids': tweet_ids}
        bearer_token = os.getenv('TWITTER_BEARER_TOKEN')  # in your .env file add TWITTER_BEARER_TOKEN=XXXXXX with your key from twitter dev account
        headers = {"Authorization": f"Bearer {bearer_token}"}
        return requests.get(url='https://api.twitter.com/2/tweets', params=params, headers=headers).json()

    def create_corresponding_tweets_df(self):
        '''
        each call is limited to 100 tweet ids, string comma seperated, and 300 calls/15 mins. We artificially time this
        so if you run calls prior to running this code it may crash (wait 15 mins)
        :return: dataset with tweet ids and tweets
        '''

        count = 0
        last_start = 0
        tweets_list = list()
        for i in range(100, self.raw_data.shape[0], 100):
            if count == 299:
                logger.info('Pausing, reached 300 requests')
                time.sleep(900)  # wait 15 mins to not overgo limit
                count = 0
            converted_list = [str(id) for id in self.raw_data[last_start:i]['tweet_id'].values]
            ids = ",".join(converted_list)
            response = self.get_tweets_call(ids)
            tweets_list.extend(response.get('data',[]))
            last_start = i
            count += 1

        tweets_pd = pd.DataFrame(tweets_list)
        tweets_pd['id'] = tweets_pd['id'].astype('int64')

        return tweets_pd

    @staticmethod
    def _get_dialect_model():
        '''
        loads and retuns the aae model
        :return:
        '''
        twitteraae_predict = Predict(vocabfile='./existing_models/twitteraae/model/model_vocab.txt',
                                     modelfile='./existing_models/twitteraae/model/model_count_table.txt')
        twitteraae_predict.load_model()

        return twitteraae_predict

    @staticmethod
    def jprint(obj):
        # create a formatted string of the Python JSON object
        text = json.dumps(obj, sort_keys=True, indent=4)
        print(text)
