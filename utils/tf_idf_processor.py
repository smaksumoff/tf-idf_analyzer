import re
import math
from typing import List, Dict, Any


class TfIdfProcessor:
    @staticmethod
    def tokenize_to_words(text: str) -> List[str]:
        """
        Tokenize the text into words:
        - Lowercase
        - Remove punctuation
        - Split into words
        :param text: Input text as a string
        :return: List of tokens (words)
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]|\d', "", text)  # Keep only alphanumeric characters and spaces
        tokens = text.split()  # Split on whitespace
        return tokens

    @staticmethod
    def tokenize_to_sentences(text: str) -> List[str]:
        """
        Tokenize the text into sentences:
        - Split sentences by '.', '!', or '?' followed by whitespace or end-of-line
        - Remove punctuation
        - Clean up leading/trailing spaces and remove any empty strings
        :param text: Input text as a string
        :return: List of sentences in the text
        """
        sentences = re.split(r'[.!?]\s*', text)
        sentences = [re.sub(r'[^\w\s]|\d', "", sentence) for sentence in sentences]  # Remove punctuation/digits
        sentences = [sentence.strip().lower() for sentence in sentences if sentence.strip()]
        return sentences

    def compute_tf(self, document: str) -> Dict[str, float]:
        """
        Compute Term Frequency (TF) for each word.
        TF = (frequency of word in document) / (total number of words in document)

        :param document: Input document as a string
        :return: Dictionary of words and their TF values
        """
        words = self.tokenize_to_words(document)
        word_counts: Dict[str, int] = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        total_words = len(words)

        tf: Dict[str, float] = {word: count / total_words for word, count in word_counts.items()}
        return tf

    def compute_idf(self, documents: str) -> Dict[str, float]:
        """
        Compute Inverse Document Frequency (IDF) for the words in a document collection.
        IDF = log(total number of documents* / number of documents containing the word)
        *Because there is only one document, every sentence in the document is treated as document

        :param documents: All sentences in the document (treated as separate documents)
        :return: Dictionary of words and their IDF values
        """
        words = set(self.tokenize_to_words(documents))  # Collect unique words
        sentences = self.tokenize_to_sentences(documents)
        total_sentences = len(sentences)
        idf: Dict[str, float] = {}

        for word in words:
            num_docs_with_word = sum(1 for sentence in sentences if word in sentence)
            idf[word] = math.log(total_sentences / (1 + num_docs_with_word))  # Add 1 to avoid division by zero
        return idf

    def compute_tfidf(self, document: str) -> List[Dict[str, Any]]:
        """
        Compute the TF-IDF table for a single document.

        :param document: Text of a single document.
        :return: List of dictionaries where each entry contains the word, TF value, and IDF value
        """
        words = self.tokenize_to_words(document)
        word_counts: Dict[str, int] = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # Compute Term Frequency (TF)
        tf = self.compute_tf(document)

        # Compute IDF across all "documents" (here, sentences in the input document)
        idf = self.compute_idf(document)

        # Combine TF and IDF to calculate TF-IDF
        tfidf_table: List[Dict[str, Any]] = [
            {
                "word": word,
                "tf": tf[word],
                "idf": round(idf.get(word, 0), 4)
            }
            for word in word_counts.keys()
        ]

        # Sort tfidf_table by IDF descending and limit to first 50
        sorted_tfidf_table: List[Dict[str, Any]] = sorted(tfidf_table, key=lambda x: x['idf'], reverse=True)[:50]
        return sorted_tfidf_table
