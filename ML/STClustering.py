import re
import math
from scipy.spatial import distance
from nltk.util import ngrams

class STClustering:
    def __init__(self, ngram_range=(1,1), max_features = 1000):
        self._ngram_range = ngram_range
        self._max_features = max_features
        self._vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, max_features=self.max_features)
        self._MC = [[{'bella': 2, 'de zio': 2, 'bella de': 2, 'zio': 2, 'de': 2, 'fratelli bella': 1, 'zio fratelli': 2, 'fratelli': 2},1,1]]

    def _tokenize(self, text):
        """"Turn text into a sequence of tokens"""
        # Replace all none alphanumeric characters with spaces
        s = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        tokens = [token for token in s.split(" ") if token != ""]
        return tokens

    def _word_ngrams(self, tokens):
        """Turn tokens into a sequence of n-grams"""
        # handle token n-grams
        min_n, max_n = self._ngram_range
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                # no need to do any slicing for unigrams
                # just iterate through the original tokens
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)

            # bind method outside of loop to reduce overhead
            tokens_append = tokens.append
            space_join = " ".join

            for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    tokens_append(space_join(original_tokens[i: i + n]))

        return tokens

    def _generate_ngrams(self, text):
        """Turns text into n-grams"""
        tokens = self._tokenize(text)
        ngrams = self._word_ngrams(tokens)
        return ngrams

    def _count_occurrences(self, ngrams, unique_ngrams):
        occurrences = {}
        for key in unique_ngrams:
            occurrences[key] = ngrams.count(key)
        return occurrences

    def _ngram_tokenizer(self, text):
        ngrams = self._generate_ngrams(text)
        unique_ngrams = set(ngrams)
        occurrences = self._count_occurrences(ngrams, unique_ngrams)
        return occurrences

    def _tf_vector(self, document):
        tf = document
        total = sum(document.values())
        for key in document.keys():
            tf[key] = tf[key]/total
        return tf

    def _idf_vector(self, document):
        idf = document
        n_docs = 1 + len(self._MC)
        for key in document.keys():
            n_valid_docs = 1 + len([1 for doc in self._MC if key in doc])
            ratio = n_valid_docs / n_docs
            idf[key] = math.log(ratio, 2)
        return idf

    def _tf_idf_vector(self, c):
        tf_vector = self._tf_vector(c)
        idf_vector = self._idf_vector(c)
        tfidf_vector = [tf_vector[ngram]*idf_vector[ngram] for ngram in c.keys()]
        return tfidf_vector

    def _cosineSimilarity(self, MCi, c):
        message_tfidf = self._tf_idf_vector(c)
        mc_tfidf = self._tf_idf_vector(MCi)
        print(message_tfidf)
        print(mc_tfidf)
        return 1 - distance.cosine(mc_tfidf, message_tfidf)

    def insert(self, message, time):
        ngrams = self._ngram_tokenizer(message)
        c = [ngrams, time, 1]
        similarities = [0 for i in range(len(self._MC))]
        self._tf_idf_vector(ngrams)
        for i in range(len(self._MC)):
            similarities[i] = self._cosineSimilarity(self._MC[i][0], ngrams)

p = STClustering(ngram_range=(1,2))
print(p.insert("bella de zio, fratelli", 1))
