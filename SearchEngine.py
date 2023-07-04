from FuzzyMatching import FuzzyMatcher
from SBertMatching import SBertMatching

class SearchEngine:
    def __init__(self, threshold=0.7, db_path='', store_path=''):
        """
        Initialize the SearchEngine object.

        Args:
            threshold (float): The threshold value for matching similarity.
            db_path (str): The path for the SBERT embeddings.
            store_path (str): The path of the CSV file used for responses.
        """
        self.db_path = db_path
        self.threshold = threshold
        self.store_path = store_path
        self.fuzzy_matcher = FuzzyMatcher()
        self.sbert_matcher = SBertMatching(db_name=db_path, store_name=store_path)
        self.threshold = threshold

    def search(self, query, list_of_strings, num_item=-1, threshold=-1):
        """
        Perform a search using both FuzzyMatcher and SBertMatching.

        Args:
            query (str): The query string.
            list_of_strings (list): The corpus of strings to match.
            num_item (int): The maximum number of matching items to retrieve. Set to -1 to retrieve all matches.
            threshold (float): The minimum similarity threshold for a match. Set to -1 to use the default threshold.

        Returns:
            list: A list of matching strings ranked by confidence.
        """
        if threshold == -1:
            threshold = self.threshold
        else:
            self.threshold = threshold

        list_of_strings = list(set(list_of_strings))
        results_fuzzy = self.fuzzy_matcher.get_matching_list(query_string=query, list_of_strings=list_of_strings,
                                                             num_item=num_item, threshold=self.threshold)
        results_sbert = self.sbert_matcher.get_matching_list(query_string=query, list_of_strings=list_of_strings
                                                             , threshold=self.threshold)
        results = self.rank_by_confidence(results_fuzzy, results_sbert)
        return results

    def rank_by_confidence(self, data1, data2):
        """
        Rank the matching strings by confidence.

        Args:
            data1 (tuple): A tuple containing the matching strings and confidence scores from FuzzyMatcher.
            data2 (tuple): A tuple containing the matching strings and confidence scores from SBertMatching.

        Returns:
            list: A list of matching strings ranked by confidence.
        """
        # Combine the two datasets
        combined_data = data1[0] + data2[0]
        combined_scores = data1[1] + data2[1]

        # Pair each statement with its confidence score
        paired_data = list(zip(combined_data, combined_scores))

        # Sort the paired data by confidence score in descending order
        paired_data.sort(key=lambda x: x[1], reverse=True)

        results = [data[0] for data in paired_data]
        return results
