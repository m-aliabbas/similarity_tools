from FuzzyMatching import FuzzyMatcher
from SBertMatching import SBertMatching
import pandas as pd
from utils import *


class Classifier:
    def __init__(self, method='Hybrid', data_file='Data/base_data.csv', text_col='Text', target_col='Disposition'):
        """
        Initialize the Classifier object.

        Args:
            method (str): The matching method to use ('fuzzy_matcher', 'sbert_matcher', 'hybrid').
            data_file (str): The file path of the base data CSV file.
            text_col (str): The column name in the CSV file that contains the text data.
            target_col (str): The column name in the CSV file that contains the target labels.
        """
        self.method = method
        self.data_file = data_file
        self.text_col = text_col
        self.target_col = target_col
        self.df = pd.read_csv(self.data_file)
        self.df = self.df.drop_duplicates(subset=[self.text_col])
        self.X = list(self.df[self.text_col])
        self.Y = list(self.df[self.target_col])
        self.fuzzy_matcher = FuzzyMatcher()
        self.sbert_matcher = SBertMatching()

    def fuzzy_predict(self, query):
        """
        Predict the target label using fuzzy matching.

        Args:
            query (str): The query string.

        Returns:
            tuple: A tuple containing the predicted target label and the matching results.
        """
        results = self.fuzzy_matcher.get_match(query, self.X)
        y_hat = list(self.df[self.df[self.text_col] == results[0]][self.target_col].values)
        return y_hat[0], results

    def sbert_predict(self, query):
        """
        Predict the target label using SBERT matching.

        Args:
            query (str): The query string.

        Returns:
            tuple: A tuple containing the predicted target label and the matching results.
        """
        results = self.sbert_matcher.get_match(query, self.X)
        y_hat = list(self.df[self.df[self.text_col] == results[0]][self.target_col].values)
        return y_hat[0], results

    def hybrid_predict(self, query, embedding_weight):
        """
        Predict the target label using a hybrid matching method.

        Args:
            query (str): The query string.
            embedding_weight (float): The weight for the SBERT matching results.

        Returns:
            tuple: A tuple containing the predicted target label and the matching results.
        """
        results1 = self.fuzzy_predict(query=query)
        results2 = self.sbert_predict(query=query)

        if results1[0] != results2[0]:
            percentage1 = results1[1][1]
            percentage2 = results2[1][1]

            # Weighting the embedding
            percentage2 *= embedding_weight

            if percentage2 < percentage1:  # Return the value of fuzzy
                return results1
            else:  # Return the results from SBERT
                return results2

        else:
            return results2

    def predict(self, query, method='', embedding_weight=1.2):
        """
        Predict the target label for the given query using the specified method.

        Args:
            query (str): The query string.
            method (str): The matching method to use ('fuzzy_matcher', 'sbert_matcher', 'hybrid').
            embedding_weight (float): The weight for the SBERT matching results in the hybrid method.

        Returns:
            str: The predicted target label.
        """
        if method in ['fuzzy_matcher', 'sbert_matcher', 'hybrid']:
            self.method = method

        if self.method == 'fuzzy_matcher':
            result = self.fuzzy_predict(query)[0]

        elif self.method == 'sbert_matcher':
            result = self.sbert_predict(query)[0]

        elif self.method == 'hybrid':
            result = self.hybrid_predict(query, embedding_weight)[0]

        else:
            print(' [-] Invalid Method; results are set to empty')
            result = ''

        return result
