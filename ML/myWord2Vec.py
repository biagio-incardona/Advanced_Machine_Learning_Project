from gensim.models.doc2vec import Doc2Vec, TaggedDocument
class myWord2Vec:
    def __init__(self, corpus, vector_size, window, min_count, n_proc, epochs):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.n_proc = n_proc
        self.epochs = epochs
        self.model = self.get_w2v_vectorizer(corpus)
        
    def tokenize(self,df):
        corpus = []
        j = 0
        for col in df:
            word_list = col.split(" ")
            word_list = ' '.join(word_list).split()
            tagged = TaggedDocument(word_list, [j])
            j = j+1
            corpus.append(tagged)
        return corpus

    def get_w2v_vectorizer(self, corpus):
        corpus = self.tokenize(corpus)
        model = Doc2Vec(vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=self.n_proc, epochs = self.epochs)
        model.build_vocab(corpus)
        model.train(corpus, total_examples=model.corpus_count
                , epochs=model.epochs)
        return model

    def text_w2v_vectorize(self, text):
        vectorized = self.model.infer_vector(text.split(' '))
        return vectorized

    def df_w2v_vectorize(self, df):
        card2vec = [self.text_w2v_vectorize(df.iloc[i])
                for i in range(0,len(df))]
        return card2vec