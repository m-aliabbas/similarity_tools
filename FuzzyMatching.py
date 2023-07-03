from fuzzywuzzy import process
from fuzzywuzzy import fuzz


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
