import faiss
import os


class FAISS:
    def __init__(self, db_name='', dims=384):
        """
        Initialize the FAISS object.

        Args:
            db_name (str): The name of the database.
            dims (int): The dimensionality of the vectors.
        """
        self.db_name = db_name
        self.dims = dims
        self.index = faiss.IndexFlatIP(self.dims)
        self.check_existence()

    def check_existence(self):
        """
        Check if the index file exists. If not, create a new index file.

        Returns:
            None
        """
        if not os.path.isfile(self.db_name + '.index'):
            faiss.write_index(self.index, self.db_name + '.index')
            self.index = faiss.read_index(self.db_name + '.index')
        else:
            self.index = faiss.read_index(self.db_name + '.index')

    def write(self):
        """
        Write the index to the index file.

        Returns:
            bool: True if successful.
        """
        faiss.write_index(self.index, self.db_name + '.index')
        return True

    def read(self):
        """
        Read the index from the index file.

        Returns:
            bool: True if successful.
        """
        self.index = faiss.read_index(self.db_name + '.index')
        return True

    def enroll(self, embd_list):
        """
        Enroll (add) embeddings to the index.

        Args:
            embd_list (numpy.ndarray): The list of embeddings to enroll.

        Returns:
            bool: True if successful.
        """
        self.index.add(embd_list)
        self.write()
        self.read()
        return True

    def refresh(self):
        """
        Refresh the index by removing the index file and creating a new index.

        Returns:
            None
        """
        if os.path.isfile(self.db_name + '.index'):
            print('[+] Refreshing Indexes')
            os.remove(self.db_name + '.index')
            self.index = faiss.IndexFlatIP(self.dims)
            self.check_existence()

    def search(self, query, num=1):
        """
        Perform a search on the index for the given query.

        Args:
            query (numpy.ndarray): The query vector.
            num (int): The number of nearest neighbors to retrieve.

        Returns:
            tuple: A tuple containing the IDs and distances of the nearest neighbors.
        """
        results = self.index.search(query, k=num)
        if num == 1:
            return results[0][0][0], results[1][0][0]
        if num > 1:
            return results[0][0], results[1][0]
        if num < 1:
            return 0, 0
