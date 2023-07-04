from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from SBertEmbedding import SBertEmbedding
from FAISS import FAISS
from utils import *

class SBertMatching:
    def __init__(self, model_path='sentence-transformers/all-MiniLM-L6-v2', db_name='base', dims=384):
        """
        Initialize the SBertMatching object.

        Args:
            model_path (str): The path or name of the pre-trained SBERT model.
            db_name (str): The name of the FAISS index database.
            dims (int): The dimensionality of the vectors.
        """
        self.model_path = model_path
        self.embeddings_model = SBertEmbedding(model_path=self.model_path)
        self.base_df = None
        self.list_of_string = []
        self.list_hash = ''
        self.indexer = FAISS(db_name=db_name, dims=dims)
        try:
            self.base_df = pd.read_csv('base.csv')
            self.list_of_string = df_to_list(input_df=self.base_df, columns=['text'])
            self.list_hash = get_object_hash(self.list_of_string)
        except FileNotFoundError:
            pass

    def enroll_list(self):
        """
        Enroll the list of strings in the FAISS index.

        Returns:
            None
        """
        text_list_embeddings = self.embeddings_model.get_embeddings(self.list_of_string)
        self.indexer.enroll(text_list_embeddings)

    def get_match(self, query_string, list_of_strings):
        """
        Get the closest matches for the given query string from the list of strings.

        Args:
            query_string (str): The query string.
            list_of_strings (list): The corpus of strings to match.

        Returns:
            tuple: A tuple containing the match and distances of the closest matches.
        """
        self.enroll_if_list_changes(list_of_strings=list_of_strings)
        query_embeddings = self.embeddings_model.get_embeddings(query_string)
        results = self.indexer.search(query_embeddings)
        return self.list_of_string[results[1]], results[0]
    
    def enroll_if_list_changes(self,list_of_strings):
        if self.list_hash != get_object_hash(list_of_strings):
            self.list_of_string = list_of_strings
            self.base_df = list_to_df(self.list_of_string, columns=['text'])
            self.list_hash = get_object_hash(self.list_of_string)
            self.base_df.to_csv('base.csv')
            self.indexer.refresh()
            self.enroll_list()

    def get_matching_list(self,query_string,list_of_strings,num_item=-1,threshold=0.3):
        self.enroll_if_list_changes(list_of_strings=list_of_strings)
        if num_item <= 0:
            num_item = len(list_of_strings)-2
        query_embeddings = self.embeddings_model.get_embeddings(query_string)
        results = self.indexer.search(query_embeddings,num=num_item)
        indexes = results[1][np.where(results[0]>threshold)]
        confidence = list(results[0][np.where(results[0]>threshold)])
        reterive_texts = [self.list_of_string[index] for index in indexes]
        return reterive_texts , confidence
