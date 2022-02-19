import re
import math
from scipy.spatial import distance
from nltk.util import ngrams

class STClustering:
    def __init__(self, ngram_range=(1,1), max_features = 1000, r=0.9, lambda_=0.01, gap_time=1):
        self._ngram_range = ngram_range
        self._max_features = max_features
        self._r = r
        self._lambda = lambda_
        self._gap_time = gap_time
        self._MC = [[{'bella': 2, 'francescano':1, 'de zio': 2, 'bella de': 2, 'zio': 2, 'de': 2, 'fratelli bella': 1, 'zio fratelli': 2, 'fratelli': 2},1,1]]

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

    def _fix_MC(self, ngrams):
        for key in ngrams:
            if key not in self._MC[0][0]:
                for ngram in self._MC:
                    ngram[0][key] = 0

    def _fix_current(self, ngrams):
        all_ngrams = set(self._MC[0][0].keys())
        ngrams = ngrams.union(all_ngrams)
        return ngrams

    def _ngram_adjust(self, ngrams):
        self._fix_MC(ngrams)
        ngrams = self._fix_current(ngrams)
        return ngrams

    def _ngram_tokenizer(self, text):
        ngrams = self._generate_ngrams(text)
        unique_ngrams = set(ngrams)
        unique_ngrams = self._ngram_adjust(unique_ngrams)
        occurrences = self._count_occurrences(ngrams, unique_ngrams)
        return occurrences

    def _tf_vector(self, document):
        tf = document.copy()
        total = sum(document.values())
        for key in document.keys():
            tf[key] = tf[key]/total
        return tf

    def _idf_vector(self, document):
        idf = document.copy()
        n_docs = 1 + len(self._MC)
        for key in document.keys():
            n_valid_docs = 1 + len([1 for doc in self._MC if key in doc])
            ratio = n_valid_docs / n_docs
            idf[key] = math.log(ratio, 2)
        return idf

    def _tf_idf_vector(self, c):
        tf_vector = c.copy()
        idf_vector = self._idf_vector(c)
        tfidf_vector = [tf_vector[ngram]*idf_vector[ngram] for ngram in c.keys()]

        return tfidf_vector

    def _cosineSimilarity(self, MCi, c):
        message_tfidf = self._tf_idf_vector(c)
        mc_tfidf = self._tf_idf_vector(MCi)
        return 1 - distance.cosine(mc_tfidf, message_tfidf)

    def _merge(self, micro_cluster, macro_cluster):
        for key in micro_cluster[0].keys():
            macro_cluster[0][key] += micro_cluster[0][key]
        macro_cluster[1] += micro_cluster[1]
        macro_cluster[2] = micro_cluster[2]

    def _get_cosine_similarities(self, ngrams):
        similarities = [0 for i in range(len(self._MC))]
        for i in range(len(self._MC)):
            similarities[i] = self._cosineSimilarity(self._MC[i][0], ngrams)
        return similarities

    def _update_weights(self, cur_time):
        for cluster in self._MC:
            decay = 2 ** (-1 * self._lambda * (cur_time - cluster[2]))
            for key in cluster[0].keys():
                cluster[0][key] = cluster[0][key]*decay
            cluster[1] = cluster[1]*decay
            cluster[2] = cur_time

    def _cleanup(self):
        pass

    def insert(self, message, time):
        ngrams = self._ngram_tokenizer(message)
        c = [ngrams, 1, time]
        similarities = self._get_cosine_similarities(ngrams)
        d = similarities.index(max(similarities))
        if similarities[d] >= self._r:
            self._merge(c, self._MC[d])
        else:
            self._MC.append(c)
        self._update_weights(time)
        if time - self._gap_time == 0:
            print("gap")
            self._cleanup()
        print(len(self._MC))
        print(similarities)

p =STClustering(ngram_range=(1,2))
p.insert("che persona del cavolo ma dai e che cazzo", 1.3)
p.insert("sei una schifo allora gne gne gne fascistissimo", 1.5)
p.insert("che persona del cavolo ma dai e che cazzo", 1.7)
p.insert("sei una schifo allora gne gne gne fascistissimo", 1.9)
p.insert("che persona del cavolo ma dai e che cazzo", 2.1)
p.insert("sei una schifo allora gne gne gne fascistissimo", 2.3)
p.insert("casiejw auasdj ldjiwqe cazzo", 2.5)
p.insert("sei una schifo allora gne gne gne fascistissimo", 2.9)