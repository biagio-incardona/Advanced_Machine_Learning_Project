import re

class Preprocess():
    def __init__(self, negations, emojis, regex_subs, stemmer):
        self.negations = negations
        self.emojis = emojis
        self.regex_subs = regex_subs
        self.stemmer = stemmer
    
    def sub_emoji(self, text):
        for emoji in self.emojis.keys():
            text = text.replace(emoji, self.emojis[emoji]) 
        return text

    def sub_negations(self, text):
        for negation in self.negations.keys():
            text = text.replace(negation, self.negations[negation])
        return text

    def sub_regexs(self, text):
        for regex in self.regex_subs.keys():
            text = re.sub(regex, self.regex_subs[regex], text)
        return text

    def stemming(self, text):
        stemmed_text = ""
        for word in text.split():
            word = self.stemmer.stem(word)
            stemmed_text += (word + " ")
        return stemmed_text

    def text_preprocess(self, text):
        text = text.lower()
        text = self.sub_emoji(text)
        text = self.sub_negations(text)
        text = self.sub_regexs(text)
        text = self.stemming(text)
        return text

    def sentiment_preprocess(self, sentiment):
        if sentiment == 4 :
            sentiment = 1
        return sentiment

    def df_pre_process(self, df, var_text, var_sentiment):
        print("starting preprocessing...")
    
        df[var_text] = df[var_text].apply(lambda x: self.text_preprocess(x))
        df[var_sentiment] = df[var_sentiment].apply(lambda x: self.sentiment_preprocess(x))
        print("...preprocessing completed")
        return df
          