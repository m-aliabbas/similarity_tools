from FuzzyMatching import FuzzyMatcher
from SBertMatching import SBertMatching
class SearchEngine:
    def __init__(self,threshold=0.7,db_path='',store_path='') -> None:
        self.db_path = db_path #path for embedding
        self.threshold = threshold
        self.store_path = store_path #path of csv file use for responses
        self.fuzzy_matcher = FuzzyMatcher()
        self.sbert_matcher = SBertMatching(db_name=db_path,store_name=store_path)
        self.threshold = threshold

    def search(self,query,list_of_strings,num_item=-1,threshold=-1):
        if threshold == -1:
            threshold = self.threshold
        else:
            self.threshold = threshold
        results_fuzzy = self.fuzzy_matcher.get_matching_list(query_string=query,list_of_strings=list_of_strings,threshold=self.threshold)
        results_sbert = self.sbert_matcher.get_matching_list(query_string=query,list_of_strings=list_of_strings,threshold=self.threshold)
        results = self.rank_by_confidence(results_fuzzy,results_sbert)
        return results
    
    def rank_by_confidence(self,data1, data2):
        # Combine the two datasets
        combined_data = data1[0] + data2[0]
        combined_scores = data1[1] + data2[1]
        
        # Pair each statement with its confidence score
        paired_data = list(zip(combined_data, combined_scores))
        
        # Sort the paired data by confidence score in descending order
        paired_data.sort(key=lambda x: x[1], reverse=True)
        
        results = [data[0] for data in paired_data]
        return results