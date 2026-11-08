from scipy.sparse import lil_matrix
from src.retrievers.tf_idf import TFIDF
from src.data_utils.document import Document
import numpy as np
from tqdm import tqdm

class BM25(TFIDF):
    def __init__(self, documents: list[dict] = None, index_path: str = None, k1: float = 1.5, b: float = 0.75) -> None:
        super(BM25, self).__init__(documents, index_path)
        self.k1 = k1
        self.b = b
    
    def GetDocumentLengths(self):
        document_lengths = {}
        average_document_length = 0
        for document in self.index.GetDocuments():
            length = len(self.PreprocessText(document.GetText()))
            document_lengths[document] = length
            average_document_length += length
        return document_lengths, average_document_length/len(self.index.GetDocuments())
    
    # TODO: Add methods for computing self.bm25_matrix


    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """
        Returns a dict with terms and their corresponding inverse document frequencies.
        """
        document_frequencies = self.GetDocumentFrequencies()
        document_length = len(self.index.GetDocuments())
        inverse_document_frequencies = {}
        for key, value in document_frequencies.items():
            inverse_document_frequencies[key] = np.log((document_length + 1) / (value + 0.5))

        return inverse_document_frequencies

    def GetDocumentsTFIDFVectors(self):
        """
        Returns a n_documents x n_terms matrix whose element (i, j) is the td-idf score of the j'th term in the i'th document.
        """
        # Create a mapping from terms to indices
        self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}

        # Initialize a sparse matrix
        n_documents = len(self.index.GetDocuments())
        n_terms = len(self.corpus_vocabulary)
        tfidf_matrix = lil_matrix((n_documents, n_terms), dtype=np.float32)
        documents = self.index.GetDocuments()
        document_idfs = self.GetInverseDocumentFrequencies()

        document_lengths, average_document_length =self.GetDocumentLengths()

        # TODO: Add tf-idf code
        for idx_doc, doc in enumerate(tqdm(documents, desc="Computing TF-IDF vectors")):
            document_term_frequencies = self.GetDocumentTermCounts(doc)
            print(f"document_term_frequencies:{document_term_frequencies} ")
            if not document_term_frequencies:
                continue


            #   continue
            max_freq = max(document_term_frequencies.values())
            for term, term_count in document_term_frequencies.items():
                idx_term =self.term_to_index[term]
                #print(document_term_frequencies)
                tf_numerator = term_count*(self.k +1)
                tf_denumerator = term_count + self.k * (1 - self.b + self.b * (document_lengths/average_document_length))
                tf =tf_numerator /tf_denumerator
                tfidf_matrix[idx_doc, idx_term] = tf*document_idfs[term]

        return tfidf_matrix.tocsr()

    
    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document
        """
        query_vector = self.QueryToVector(queries)
        scores = query_vector.dot(self.bm25_matrix.T).toarray()
        return scores