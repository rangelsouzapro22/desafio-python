import json
import itertools
import os
import string
import sys
import swifter
from collections import defaultdict
from itertools import groupby
from os import listdir
from os.path import isfile, join
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
STOP_WORDS = stopwords.words()
BASEPATH = Path(os.path.normpath(os.getcwd() + os.sep + os.pardir))

sys.path.append(BASEPATH / 'src')

from utilities import highlight_search_word


class Collection:

    def __init__(self):
        self.base = defaultdict(dict)
        return

    def read(self, doc_id, row_id):
        """
        Read collection database parsed by (doc_id, row_id)

        :param doc_id(str): document identification
        :param row_id(str): row number indentificantion into de the document
        :return(str): content of specif row in the document
        """
        row = self.base.get(str(doc_id))
        if isinstance(row, dict):
            return row.get(str(row_id))

    def add(self, doc_id, row_id, content):
        """
        update content by identification's keys
        :param doc_id(str):
        :param row_id(str):
        :param content(str):
        :return:
        """
        self.base[doc_id].update({row_id: content})

    def delete(self, key):
        """
        Delete document in collection database
        :param key (str): document identification
        :return:
        """
        self.base.pop(key, None)


class ReverseIndex:

    def __init__(self):
        self._path_reverse_file = BASEPATH / 'output' / 'reverse_index.json'
        self._path_collection_file = BASEPATH / 'output' / 'collection.json'
        self.index = defaultdict(list)
        self.collection = Collection()
        self.initialize()

    def initialize(self):
        """
        Process for initialize reverse index database and content database.
        """
        if not os.path.exists(self._path_reverse_file):
            print('creating new reverse index collection (~16min)')
            self.processing()
            return
        print(150 * '-')
        print('reading reverse index collection from backup')
        self.start_index()
        self.start_collection()

    def start_index(self):
        """
        Load reverse index data into attribute's list from backup.
        """
        with open(self._path_reverse_file) as f:
            self.index = json.load(f)
        return

    def start_collection(self):
        """
        Load content data into attribute's list from backup.
        """
        with open(self._path_collection_file) as f:
            self.collection.base = json.load(f)
        return

    @staticmethod
    def pre_processing(content):
        """
        Analizing data to remove punctuation and get the word's stemming

        :param content (str): content row from document
        :return (list): list of stem in content desconsidering stop words
        """
        content_wt_punc = content.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(content_wt_punc.lower())
        _stemmer = SnowballStemmer("english", ignore_stopwords=True)
        stems_tokens = [_stemmer.stem(wr) for wr in tokens]
        return list(set(stems_tokens) - set(STOP_WORDS))

    @staticmethod
    def get_frequency(doc_id, row_id, tokens):
        """
        Calculate word's frequency in token's list and inverting index by tokens

        :param doc_id (str): document identification
        :param row_id (str): row of document identification
        :param tokens (list): list of stems
        :return:
        """
        return {key: {'doc_id': doc_id, 'row_id': row_id} for key, group in groupby(tokens)}#, 'freq': len(list(group))

    def search_word(self, sentence):
        """
        Search the stem of word(s) on reverse index database and select the respectively content.

        :param sentence (str): text input from user
        :return (dict): collection of document where the tokens appear
        """
        tokens = self.pre_processing(sentence)
        result_search = {token: self.index[token] for token in tokens if token in self.index}
        tmp = [result_search[token] for token in result_search.keys()]
        tmp = list(itertools.chain.from_iterable(tmp))
        unique_rows = sorted([dict(y) for y in set(tuple(x.items()) for x in tmp)]
                             , key=lambda i: (int(i['doc_id']), int(i['row_id'])))
        print(150 * '-')
        print(f"{len(unique_rows)} results")
        print(150 * '-')
        for idx, item in enumerate(unique_rows):
            content = self.collection.read(item['doc_id'], item['row_id'])
            print(highlight_search_word(item['doc_id'], item['row_id'], tokens, content))
            if idx > 0 and idx % 100 == 0:
                _ = input('Press Enter to see the next 100 results')
        return unique_rows

    @staticmethod
    def load_file(file_id, path_file):
        """
        Read the content of file
        :param file_id (str): document identification
        :param path_file (Path): path of file
        :return (pandas.DataFrame): frame of content
        """
        content = open(path_file, 'r')
        read = content.read()
        array_content = read.split('\n')
        df = pd.DataFrame({'content': array_content})
        df['doc_id'] = file_id
        df = df.reset_index().rename(columns={'index': 'row_id'})
        return df

    def load_documents(self):
        """
        Load and pre processing the content from all files.
        :return (dict): collection of content available rows
        """
        path = BASEPATH / 'dataset'
        files = [f for f in listdir(path) if isfile(join(path, f))]
        files.sort()
        frames = []
        for file in files:
            df = self.load_file(file_id=file, path_file=path / file)
            frames.append(df)
        df_content = pd.concat(frames)
        df_content.reset_index(drop=True, inplace=True)
        df_content = df_content[df_content['content'] != '']
        df_content['tokens'] = df_content['content'].swifter.set_npartitions(8).apply(lambda x: self.pre_processing(x))
        return df_content.to_dict(orient='records')

    def __backup_result(self):
        """
        create backup files for reverse index database and collection content
        """
        dict_files = {'reverse_index.json': self.index, 'collection.json': self.collection.base}
        for key in dict_files:
            with open(BASEPATH / 'output' / key, "w") as outfile:
                json.dump(dict_files[key], outfile)

    def processing(self):
        """
        main process responsible to control the calls
        """
        start = pd.to_datetime('now')
        collection = self.load_documents()
        for item in collection:
            freq_tokens = self.get_frequency(doc_id=item['doc_id'], row_id=item['row_id'], tokens=item['tokens'])
            for word, data in freq_tokens.items():
                self.index[word].append(data)
            self.collection.add(doc_id=item['doc_id'], row_id=item['row_id'], content=item['content'])
        print('loading time: ', pd.to_datetime('now') - start)
        self.__backup_result()

    def run(self):
        """
        user interface to recieve word's search
        :return:
        """
        while True:
            print(150 * '*')
            search_term = input("Enter your sentence to search or (q) to quit: ")
            if search_term == 'q':
                break
            _ = self.search_word(search_term)
            input("Enter any button to continue...")
            os.system('cls')


if __name__ == '__main__':
    _reverse_index = ReverseIndex()
    _reverse_index.run()



