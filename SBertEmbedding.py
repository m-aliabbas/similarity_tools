from sentence_transformers import SentenceTransformer
import numpy as np


class SBertEmbedding:
    def __init__(self, model_path='sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize the SBertEmbedding object.

        Args:
            model_path (str): The path or name of the pre-trained SBERT model.
        """
        self.model_path = model_path
        self.model = SentenceTransformer(self.model_path)

    def get_embeddings(self, text):
        """
        Get the sentence embeddings for the given text.

        Args:
            text (str or list): The input text or list of texts.

        Returns:
            numpy.ndarray: The sentence embeddings.
        """
        embeddings = self.model.encode(text)
        if len(embeddings.shape) <= 1:
            embeddings = np.expand_dims(embeddings, axis=0)
        return embeddings
