from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from SBertEmbedding import SBertEmbedding
from FAISS import FAISS
from utils import *


class SBertMatching:
    def __init__(self, model_path='sentence-transformers/all-MiniLM-L6-v2', db_name='base',
                 store_name='base.csv', dims=384):
        """
        Initialize the SBertMatching object.

        Args:
            model_path (str): The path or name of the pre-trained SBERT model.
            db_name (str): The name of the FAISS index database.
            store_name (str): The name of the file to store the base dataframe.
            dims (int): The dimensionality of the vectors.
        """
        self.model_path = model_path
        self.embeddings_model = SBertEmbedding(model_path=self.model_path)
        self.base_df = None
        self.list_of_strings = []
        self.list_hash = ''
        self.indexer = FAISS(db_name=db_name, dims=dims)
        self.store_name = store_name

        try:
            # Try loading the base dataframe from the file
            self.base_df = pd.read_csv(store_name)
            self.list_of_strings = df_to_list(input_df=self.base_df, columns=['text'])
            self.list_hash = get_object_hash(self.list_of_strings)
        except FileNotFoundError:
            pass

    def enroll_list(self):
        """
        Enroll the list of strings in the FAISS index.

        Returns:
            None
        """
        text_list_embeddings = self.embeddings_model.get_embeddings(self.list_of_strings)
        self.indexer.enroll(text_list_embeddings)

    def get_match(self, query_string, list_of_strings):
        """
        Get the closest matches for the given query string from the list of strings.

        Args:
            query_string (str): The query string.
            list_of_strings (list): The corpus of strings to match.

        Returns:
            tuple: A tuple containing the matches and distances of the closest matches.
        """
        list_of_strings = list(set(list_of_strings))
        self.enroll_if_list_changes(list_of_strings=list_of_strings)
        query_embeddings = self.embeddings_model.get_embeddings(query_string)
        results = self.indexer.search(query_embeddings)
        matches = self.list_of_strings[results[1]]
        distances = results[0]
        return matches, distances

    def enroll_if_list_changes(self, list_of_strings):
        """
        Enroll the list of strings if there are changes since the last enrollment.

        Args:
            list_of_strings (list): The updated corpus of strings.

        Returns:
            None
        """
        if self.list_hash != get_object_hash(list_of_strings):
            print(len(self.list_of_strings),len(list_of_strings))
            self.list_of_strings = list_of_strings
            self.base_df = list_to_df(self.list_of_strings, columns=['text'])
            self.list_hash = get_object_hash(self.list_of_strings)
            self.base_df.to_csv(self.store_name)
            self.indexer.refresh()
            self.enroll_list()

    def get_matching_list(self, query_string, list_of_strings, num_items=-1, threshold=0.3):
        """
        Get a list of matching strings based on a query string.

        Args:
            query_string (str): The query string.
            list_of_strings (list): The corpus of strings to match.
            num_items (int): The maximum number of matching items to retrieve. Set to -1 to retrieve all matches.
            threshold (float): The minimum similarity threshold for a match.

        Returns:
            tuple: A tuple containing the matching strings and confidence scores.
        """
        self.enroll_if_list_changes(list_of_strings=list_of_strings)
        
        if num_items <= 0:
            num_items = len(list_of_strings) - 2

        query_embeddings = self.embeddings_model.get_embeddings(query_string)
        results = self.indexer.search(query_embeddings, num=num_items)
        indexes = results[1][np.where(results[0] > threshold)]
        confidence = list(results[0][np.where(results[0] > threshold)])
        matching_strings = [self.list_of_strings[index] for index in indexes]
        
        return matching_strings, confidence
