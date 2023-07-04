from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import numpy as np

class FuzzyMatcher:
    def __init__(self, threshold=0.0):
        """
        Initialize the FuzzyMatcher object.

        Args:
            threshold (float): The threshold value for matching similarity.
        """
        self.threshold = threshold
        self.results = ('', 0)  # matched string, distance

    def find_match(self, query_string, list_of_strings):
        """
        Find the closest matching string based on Levenshtein Distance.

        Args:
            query_string (str): The input string.
            list_of_strings (list): The corpus of strings to match.

        Returns:
            str: The matched string.
        """
        self.results = process.extractOne(query_string, list_of_strings)
        return self.results
    
    def get_matching_list(self, query_string, list_of_strings, num_item=-1, threshold=0.3):
        """
        Get a list of matching strings based on a query string.

        Args:
            query_string (str): The query string.
            list_of_strings (list): The corpus of strings to match.
            num_item (int): The maximum number of matching items to retrieve. Set to -1 to retrieve all matches.
            threshold (float): The minimum similarity threshold for a match.

        Returns:
            tuple: A tuple containing the matching strings and confidence scores.
        """
        if num_item <= 0:
            num_item = len(list_of_strings) - 1
        else:
            threshold = 0.0
        
        results = process.extract(query_string, list_of_strings, limit=num_item)
        confidence = [fuzz.token_sort_ratio(query_string, result[0]) / 100 for result in results]
        indexes = np.where(np.array(confidence) > threshold)[0]
        conf_sent_back = [confidence[index] for index in indexes]
        retrieve_texts = [list_of_strings[index] for index in indexes]
        
        return retrieve_texts, conf_sent_back

    def find_similarity_ratio(self, query_string, result_string):
        """
        Calculate the similarity ratio between two strings.

        Args:
            query_string (str): The query string.
            result_string (str): The result string to compare.

        Returns:
            float: The similarity ratio.
        """
        token_sort_ratio = fuzz.token_sort_ratio(query_string, result_string)
        distance = self.results[1]
        similarity_ratio = token_sort_ratio / distance
        return similarity_ratio

    def get_match(self, query_string, list_of_strings):
        """
        Get the best match for the given query string from the list of strings.

        Args:
            query_string (str): The query string.
            list_of_strings (list): The corpus of strings to match.

        Returns:
            tuple: A tuple containing the matched string and its similarity ratio.
        """
        results = self.find_match(query_string=query_string, list_of_strings=list_of_strings)
        similarity_ratio = self.find_similarity_ratio(query_string, results[0])
        self.results = (results[0], similarity_ratio)
        return self.results
