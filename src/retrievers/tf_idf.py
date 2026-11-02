from src.data_utils.document import Document
from src.retrievers.base_retriever import Retriever
import numpy as np
from scipy.sparse import lil_matrix
import re
from collections import Counter


class TFIDF(Retriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None) -> None:
        super(TFIDF, self).__init__(documents, index_path)
        self.corpus_vocabulary = self.GetCorpusVocabulary()
        self.idf = self.GetInverseDocumentFrequencies()

        # Only run if we are in TFIDF class
        if type(self) == TFIDF:
            self.tfidf_matrix = self.GetDocumentsTFIDFVectors()

    def PreprocessText(self, text: str) -> list[str]:
        """
        Remove unwanted charatcters from text and lowercase
        """
        #raise NotImplementedError()
        lowercase_text = text.lower()
        lowercase_text = re.sub(r'[^a-z0-9 ]', "", lowercase_text)
        split_text = lowercase_text.split()
        return split_text
        
    def GetQueryVocabulary(self, query: str) -> list[str]:
        """
        Returns all unique words in a query
        """
        #raise NotImplementedError()
        unique_words_query = list(set(self.PreprocessText(query)))
        return unique_words_query

    def GetDocumentVocabulary(self, document: Document) -> list[str]:
        """
        Returns all unique words in a document
        """
        #raise NotImplementedError()
        unique_words_document = list(set(self.PreprocessText(document.GetText())))
        return unique_words_document
    
    def GetCorpusVocabulary(self) -> set[str]:
        """
        Returns all unique words in ALL documents.
        """
        # Use map function for more efficient processing
        processed_texts = map(lambda doc: set(self.PreprocessText(doc.GetText())), self.index.GetDocuments())
        corpus_vocabulary = set().union(*processed_texts)
        return corpus_vocabulary

    def GetDocumentFrequencies(self) -> dict[str, int]:
        """
        Returns a dict with terms and counts of how many documents the term occurs in (counted once per document).
        """
        #raise NotImplementedError()
        documents_list = self.index.GetDocuments()
        huge_word_list = []
        for document in documents_list:
            document_words = self.GetDocumentVocabulary(document)
            huge_word_list.extend(document_words)

        counter = Counter(huge_word_list)
        most_common = counter.most_common()
        document_frequencies = dict(most_common)
        return document_frequencies

       
    
    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """
        Returns a dict with terms and their corresponding inverse document frequencies.
        """
        #raise NotImplementedError()
        document_frequencies = self.GetDocumentFrequencies()
        document_length = len(self.index.GetDocuments())
        inverse_document_frequencies = {}
        for key, value in document_frequencies.items():
            inverse_document_frequencies[key] = np.log(document_length/value)

        return inverse_document_frequencies
    
    
    def GetDocumentTermCounts(self, document: Document):
        """
        Returns a dict on the form {term: num_occurences} for an input Document.
        """
        #raise NotImplementedError()
        document_words = self.PreprocessText(document.GetText())

        counter = Counter(document_words)
        most_common = counter.most_common()
        document_term_frequencies = dict(most_common)
        return document_term_frequencies


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
        terms = self.corpus_vocabulary
        # TODO: Add tf-idf code
        for idx_doc, doc in enumerate(documents):
            for idx_term, term in enumerate(terms):
                tf = 
        
        return tfidf_matrix.tocsr()   # Convert to CSR format for efficient row slicing
    
    def QueryToVector(self, queries: list[str]):
        """ Converts a query or batch of queries into a sparse vector matching the vocab size with counts of how many times the term occurs.
        
        Args:
            query (str): The query to convert
        
        Returns:
            csr_matrix: A sparse vector representation of the query
        """
        query_vector = lil_matrix((len(queries), len(self.corpus_vocabulary)), dtype=np.float32)

        # TODO: add code

        return query_vector.tocsr()

    #@time_func
    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document

            num_queries x num_documents where (i, j) = ∑_i TF-IDF(q_i, D_j; D) 
        
        """
        # Convert the queries to a vector
        query_vectors = self.QueryToVector(queries) # n_queries x vocab size
        scores = query_vectors.dot(self.tfidf_matrix.T).toarray() # n_queries x n_documents
        return scores
    