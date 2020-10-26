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

OUTPUT_DIRECTORY = '../../data'

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
