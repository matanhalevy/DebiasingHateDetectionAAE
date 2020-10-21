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
import xml.etree.ElementTree as et

import requests

from ..existing_models.twitteraae.code.predict import Predict
from sklearn.model_selection import train_test_split

import logging
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)

OUTPUT_DIRECTORY = '../../data'


# TODO: use path_to_raw instead of hardcoding
# TODO: Don't prepare data if cleaned data already exists


# This assumes we don't need to drop columns that won't be used for modelling, remove prior to saving otherwise.
class DataPreparer:
    def __init__(self, path_to_raw, verbose=False):
        self.path_to_raw = path_to_raw
        self.verbose = verbose
        self.raw_data = None
        self.clean_data = None
        self.path_to_save_cleaned = None

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
        self.clean_data.to_csv(f'{OUTPUT_DIRECTORY}/{self.path_to_save_cleaned}')

    def append_is_aae(self, cleaned_df, text_label):
        # Output proportions are for the African-American, Hispanic, Asian, and White topics for text
        # recommended to assign dialict labels to tweets with dialect probabilities > 80%
        dialect_model = self._get_dialect_model()

        cleaned_df['dialect_prs'] = cleaned_df.apply(
            lambda row: dialect_model.predict(row[text_label]), axis=1)
        cleaned_df['is_aae'] = cleaned_df['dialect_prs'].str[0] >= 0.8
        cleaned_df['is_aae'] = cleaned_df['is_aae'].astype(int)

        return cleaned_df

    def _create_test_train_dev_splits(self, tr_size=0.8, dev_test_split=0.5, stratify_label='is_hate'):
        train, valid_test = train_test_split(self.clean_data, train_size=tr_size, random_state=42,
                                             stratify=self.clean_data[[stratify_label]])
        test, valid = train_test_split(valid_test, train_size=dev_test_split, random_state=42,
                                       stratify=valid_test[['target_pop']])

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
    def _get_dialect_model():
        '''
        loads and retuns the aae model
        :return:
        '''
        twitteraae_predict = Predict(vocabfile='../existing_models/twitteraae/model/model_vocab.txt',
                                     modelfile='../existing_models/twitteraae/model/model_count_table.txt')
        twitteraae_predict.load_model()

        return twitteraae_predict

    @staticmethod
    def jprint(obj):
        # create a formatted string of the Python JSON object
        text = json.dumps(obj, sort_keys=True, indent=4)
        print(text)


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
        self.path_to_save_cleaned = 'cleaned_gab/gab_cleaned_with_aae.csv'

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
                          path_or_buf=f'{OUTPUT_DIRECTORY}/cleaned_gab/majority_gab_dataset_25k/train.jsonl')
        gab_test.to_json(orient='records', lines=True, index=True,
                         path_or_buf=f'{OUTPUT_DIRECTORY}/cleaned_gab/majority_gab_dataset_25k/test.jsonl')
        gab_valid.to_json(orient='records', lines=True, index=True,
                          path_or_buf=f'{OUTPUT_DIRECTORY}/cleaned_gab/majority_gab_dataset_25k/dev.jsonl')


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

    def save_test_train_dev_splits(self):
        # gab stratified on target pop. 24353 train, 1716 valid, 1586 test, 88/0.062/0.058
        stormfront_train, stormfront_valid, stormfront_test = self._create_test_train_dev_splits(tr_size=0.88,
                                                                                                 dev_test_split=0.5,
                                                                                                 stratify_label='is_hate')
        # train: 861 7879 0.1093%   test: 214 1962 0.1091    dev: 121 1103 0.1097%

        stormfront_train.to_csv(f'{OUTPUT_DIRECTORY}/white_supremacy/train.tsv', sep='\t')
        stormfront_valid.to_csv(f'{OUTPUT_DIRECTORY}/white_supremacy/dev.tsv', sep='\t')
        stormfront_test.to_csv(f'{OUTPUT_DIRECTORY}/white_supremacy/test.tsv', sep='\t')


class NewYorkTimesArticlePrepaper(DataPreparer):
    ''''
    data source: https://catalog.ldc.upenn.edu/LDC2008T19
    API: https://developer.nytimes.com/faq
    from January 1, 1987 and June 19, 2007
    '''

    def __init__(self, path_to_raw, verbose=False):
        super().__init__(path_to_raw, verbose)
        self.path_to_raw = f'{OUTPUT_DIRECTORY}/nyt_corpus_final_data/final_df.csv'
        self.path_to_save_cleaned = 'nyt_corpus_final_data/final_df_uniform_sample.csv'
        self.extracted_files_path = f'{OUTPUT_DIRECTORY}/nyt_corpus/extracted_data/'
        self.extracted_files_path_tar = f'{OUTPUT_DIRECTORY}/nyt_corpus/tar_data/'
        self.complete_dataset_path = f'{OUTPUT_DIRECTORY}/nyt_corpus/final_data/final_df.csv'
        self.identity_terms = ['muslim', 'jew', 'jews', 'white', 'islam', 'blacks', 'muslims', 'women', 'whites', 'gay',
                               'black', 'democrat', 'islamic', 'allah', 'jewish', 'lesbian', 'transgender', 'race',
                               'brown', 'woman', 'mexican', 'religion', 'homosexual', 'homosexuality', 'africans']

    def prepare_data(self):
        if os.path.exists(self.path_to_raw):
            self.raw_data = pd.read_csv(self.path_to_raw)
        else:
            self._untargz_nyt_corpus()
            self._filter_articles_containing_identifiers()  # this takes around 12 hours on my computer
            self._create_final_df_from_files()  # this took about 2 weeks
        if os.path.exists(f'{OUTPUT_DIRECTORY}/{self.path_to_save_cleaned}'):
            self.raw_data = pd.read_csv(f'{OUTPUT_DIRECTORY}/{self.path_to_save_cleaned}')
        else:
            self._uniformly_distribute_samples()

    def _filter_articles_containing_identifiers(self):
        col_identity_terms = self.identity_terms + ['full_text']
        year_files = os.listdir(self.extracted_files_path)
        for year in year_files:
            months = os.listdir(f'{self.extracted_files_path}{year}/')
            for month in months:
                article_dict = dict()
                days = os.listdir(f'{self.extracted_files_path}{year}/{month}/')
                for day in days:
                    files = os.listdir(f'{self.extracted_files_path}{year}/{month}/{day}/')
                    for file in files:
                        self._check_if_article_contains_identifier(article_dict,
                                                                   f'{self.extracted_files_path}{year}/{month}/{day}/{file}')
                month_df = pd.DataFrame.from_dict(article_dict, orient='index', columns=col_identity_terms)
                logger.info(f'Saving {year}, {month} df, shape:', month_df.shape)
                month_df.to_csv(f'{OUTPUT_DIRECTORY}/nyt_corpus/filtered_data/identifier_articles_{year}_{month}.csv')

    def _create_final_df_from_files(self):
        nyt_all_df = pd.DataFrame(columns=['nyt_text', 'hate', 'keyword'])

        for filtered_file in os.listdir(f'{OUTPUT_DIRECTORY}/nyt_corpus/filtered_data/'):
            logger.info(filtered_file)
            temp_df = pd.read_csv(f'{OUTPUT_DIRECTORY}/nyt_corpus/filtered_data/{filtered_file}')
            for _, row in temp_df.iterrows():
                nyt_all_df = self.create_final_df(nyt_all_df, row)

        nyt_all_df.to_csv(self.complete_dataset_path)
        self.raw_data = nyt_all_df
        return nyt_all_df

    def _uniformly_distribute_samples(self):
        counts_of_nyt_identifiers = self.raw_data.groupby('keyword').agg('count').sort_values("hate", ascending=True)
        logger.info(counts_of_nyt_identifiers)

        # note that the minimum sentences with an identifier is 'transgender' with 552 instances
        uniform_sample_nyt = pd.DataFrame(columns=['nyt_text', 'hate', 'keyword'])
        min_n = counts_of_nyt_identifiers.iloc[0, 0]
        for identifier in self.identity_terms:
            temp_df = self.raw_data[self.raw_data['keyword'] == identifier]
            sample_df = temp_df.sample(n=min_n, random_state=42, replace=False)
            uniform_sample_nyt = uniform_sample_nyt.append(sample_df)

        uniform_sample_nyt = uniform_sample_nyt.sample(frac=1, random_state=42)
        self.clean_data = uniform_sample_nyt
        self.save_cleaned_data()

    def _create_final_df(self, df, df_row):
        in_article = [col for col in self.identity_terms if df_row[col] == 1]
        article_sentences = self.split_into_sentences(df_row['full_text'])
        for sentence in article_sentences:
            for identifier in in_article:
                if identifier in sentence and re.search(r"\b" + re.escape(identifier) + r"\b", sentence):
                    df = df.append({'nyt_text': sentence, 'hate': 0, 'keyword': identifier}, ignore_index=True)

        return df

    def _check_if_article_contains_identifier(self, article_dict: dict, path: str) -> None:
        xtree = et.parse(path)
        xroot = xtree.getroot()
        search = xroot.findall(".//body/body.content/*[@class='full_text']")
        full_text = ""
        for node in search:
            for para in node:
                full_text += para.text
                full_text += '\n'

        full_text = full_text.lower()
        identifiers_found = list()
        for identifier in self.identity_terms:
            if self.contains_str(full_text, identifier):
                identifiers_found.append(1)
            else:
                identifiers_found.append(0)
        if sum(identifiers_found) > 0:
            identifiers_found.append(full_text)
            article_dict[path] = identifiers_found

    def _untargz_nyt_corpus(self):
        year_files = os.listdir(f'{OUTPUT_DIRECTORY}/nyt_corpus/data')

        for year in year_files:
            months_zipped = os.listdir(f'{OUTPUT_DIRECTORY}/nyt_corpus/data/{year}/')
            new_location = self.extracted_files_path + year
            for month in months_zipped:
                self._untargzirator(f'{OUTPUT_DIRECTORY}/nyt_corpus/data/{year}/{month}', new_location)
                logger.info(new_location)

    @staticmethod
    def _untargzirator(fname, new_location):
        Path(new_location).mkdir(parents=True, exist_ok=True)
        if fname.endswith(".tgz"):
            tar = tarfile.open(fname, "r")
            tar.extractall(new_location)
            tar.close()
        elif fname.endswith("tar"):
            tar = tarfile.open(fname, "r:")
            tar.extractall(new_location)
            tar.close()

    @staticmethod
    def _contains_str(body, word) -> bool:
        if word in body and re.search(r"\b" + re.escape(word) + r"\b", body):
            return True
        else:
            return False

    @staticmethod
    def split_into_sentences(text):
        # All text is lower case https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences
        # -*- coding: utf-8 -*-
        alphabets = "([A-Za-z])"
        prefixes = "(mr|st|mrs|ms|dr)[.]"
        suffixes = "(inc|ltd|jr|sr|co)"
        starters = "(mr|mrs|ms|dr|he\s|she\s|it\s|they\s|their\s|our\s|we\s|but\s|however\s|that\s|this\s|wherever)"
        acronyms = "([a-z][.][a-z][.](?:[a-z][.])?)"
        websites = "[.](com|net|org|io|gov)"
        digits = "([0-9])"

        text = " " + text + "  "
        text = text.replace("\n", " ")
        text = re.sub(digits + "[.]" + digits, "\\1<prd>\\2", text)
        text = re.sub(prefixes, "\\1<prd>", text)
        text = re.sub(websites, "<prd>\\1", text)
        if "Ph.D" in text: text = text.replace("ph.d.", "ph<prd>d<prd>")
        text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
        text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
        text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
        text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
        text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
        if "”" in text: text = text.replace(".”", "”.")
        if "\"" in text: text = text.replace(".\"", "\".")
        if "!" in text: text = text.replace("!\"", "\"!")
        if "?" in text: text = text.replace("?\"", "\"?")
        text = text.replace(".", ".<stop>")
        text = text.replace("?", "?<stop>")
        text = text.replace("!", "!<stop>")
        text = text.replace("<prd>", ".")
        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences]
        return sentences


class TwitterDavidsonPreparer(DataPreparer):
    """
    Davidson:
    count = number of CrowdFlower users who coded each tweet (min is 3, sometimes more users coded a tweet when judgments were determined to be unreliable by CF).

    hate_speech = number of CF users who judged the tweet to be hate speech.
    offensive_language = number of CF users who judged the tweet to be offensive.
    neither = number of CF users who judged the tweet to be neither offensive nor non-offensive.
    class = class label for majority of CF users. 0 - hate speech 1 - offensive language 2 - neither
    """

    def __init__(self, path_to_raw, verbose=False):
        super().__init__(path_to_raw, verbose)
        self.path_to_save_cleaned = 'twitter_datasets/cleaned/davidson_cleaned.csv'

    def load_data(self):
        self.raw_data = pd.read_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/davidson/davidson_labelled.csv')
        if self.verbose:
            logger.info(f'Uncleaned Twitter Davidson Shape: {self.raw_data.shape}')

    def prepare_data(self, append_aae=True):
        self.raw_data = pd.read_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/davidson_labelled.csv', index_col=0)
        twitter_df = self.raw_data
        twitter_df['cleaned_tweet'] = twitter_df.apply(lambda row: self.preprocess_twitter_davidson(row['tweet']),
                                                       axis=1)
        twitter_df['is_hate'] = np.where(twitter_df['class'] == 0, 1, 0)
        twitter_df['is_offensive'] = np.where(twitter_df['class'] == 1, 1, 0)
        if append_aae:
            twitter_df = self.append_is_aae(twitter_df, 'cleaned_tweet')

        self.clean_data = twitter_df
        self.save_cleaned_data()

    def save_test_train_dev_splits(self):
        twitter_train, twitter_valid, twitter_test = self._create_test_train_dev_splits(tr_size=0.8,
                                                                                        dev_test_split=0.5,
                                                                                        stratify_label='is_hate')

        twitter_train.to_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/davidson/train.csv')
        twitter_valid.to_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/davidson/dev.csv')
        twitter_test.to_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/davidson/test.csv')


class FountaTwitterPreparer(DataPreparer):
    '''
    Founta et al:
    80k
    - literature link: https://datalab.csd.auth.gr/wp-content/uploads/publications/17909-77948-1-PB.pdf
    - dataset Link: https://dataverse.mpi-sws.org/dataset.xhtml?persistentId=doi:10.5072/FK2/ZDTEMN
    'abusive', 'normal', 'hateful', 'spam' & nan as maj_label
    '''
    def __init__(self, path_to_raw, verbose=False):
        super().__init__(path_to_raw, verbose)
        self.path_to_save_cleaned = 'twitter_datasets/cleaned/founta_cleaned.csv'

    def load_data(self):
        self.raw_data = pd.read_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/founta/hatespeechtwitter.csv')
        if self.verbose:
            logger.info(f'Uncleaned Twitter Founta Shape: {self.raw_data.shape}')

    def prepare_data(self, append_aae=True):
        self.raw_data = pd.read_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/founta/hatespeechtwitter.csv')
        self.raw_data.dropna(inplace=True)  # 4 instances
        self.raw_data['is_hate'] = np.where(self.raw_data['maj_label'] == 'hateful', 1, 0)
        self.raw_data['is_abusive'] = np.where(self.raw_data['maj_label'] == 'abusive', 1, 0)

        if os.path.exists(f'{OUTPUT_DIRECTORY}/twitter_datasets/founta/with_tweets.csv'):
            founta_twitter = pd.read_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/founta/with_tweets.csv')
        else:
            founta_tweets_df = self.create_corresponding_tweets_df()
            # 79996 rows originally, only able to retrieve 50398 tweets
            founta_twitter = self.raw_data.merge(founta_tweets_df[['id', 'text']], left_on='tweet_id', right_on='id', how='left')
            founta_twitter.dropna(inplace=True)
            founta_twitter.drop(columns=['id'])
            founta_twitter.to_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/founta/with_tweets.csv')

        founta_twitter['cleaned_tweet'] = founta_twitter.apply(lambda row: self.preprocess_twitter_davidson(row['text']), axis=1)
        if append_aae:
            founta_twitter = self.append_is_aae(founta_twitter, 'cleaned_tweet')

        self.clean_data = founta_twitter
        self.save_cleaned_data()

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
        tweets_founta = list()
        for i in range(100, self.raw_data.shape[0], 100):
            if count == 299:
                logger.info('Pausing, reached 300 requests')
                time.sleep(900)  # wait 15 mins to not overgo limit
                count = 0
            converted_list = [str(id) for id in self.raw_data[last_start:i]['tweet_id'].values]
            ids = ",".join(converted_list)
            response = self.get_tweets_call(ids)
            tweets_founta.extend(response['data'])
            last_start = i
            count += 1

        founta_tweets_pd = pd.DataFrame(tweets_founta)
        founta_tweets_pd['id'] = founta_tweets_pd['id'].astype('int64')

        return founta_tweets_pd

    def save_test_train_dev_splits(self):
        twitter_train, twitter_valid, twitter_test = self._create_test_train_dev_splits(tr_size=0.8,
                                                                                        dev_test_split=0.5,
                                                                                        stratify_label='is_hate')
        # train: 861 7879 0.1093%   test: 214 1962 0.1091    dev: 121 1103 0.1097%

        twitter_train.to_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/founta/train.csv')
        twitter_valid.to_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/founta/dev.csv')
        twitter_test.to_csv(f'{OUTPUT_DIRECTORY}/twitter_datasets/founta/test.csv')