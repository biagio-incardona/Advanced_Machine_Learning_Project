from settings import BASE_DIR
from nltk.stem import SnowballStemmer
import Preprocess as ps
from gensim.parsing.preprocessing import remove_stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from model_selection import load_dataset


class Kmeanclustering:

    def tfvec(df, text):
        vectorizer_ntf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
        X_ntf = vectorizer_ntf.fit_transform(text)
        return X_ntf

    def main(self=None):
        path = "{BaseDir}/Advanced_Machine_Learning_Project/data/dataset.csv".format(BaseDir=BASE_DIR)
        columns = ["sentiment", "ids", "date", "flag", "user", "text"]
        final_columns = ["text", "sentiment"]

        # explicit negations
        negations = {"isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
                     "haven't": "have not", "hasn't": "has not", "hadn't": "had not", "won't": "will not",
                     "wouldn't": "would not", "don't": "do not", "doesn't": "does not", "didn't": "did not",
                     "can't": "can not", "couldn't": "could not", "shouldn't": "should not", "mightn't": "might not",
                     "mustn't": "must not"}

        # convert twitter emojis in twitch style emojis
        emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
                  ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
                  ':-@': 'shocked', ':@': 'shocked', ':-$': 'confused',
                  ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
                  '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
                  '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
                  ';-)': 'wink', 'O:-)': 'angel', 'O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'}

        regex_subs = {
            r"https?://[^s]+": "URL",  # replace any url with URL
            "www.[^ ]+": "URL",  # replace any url with URL
            r"@[^\s]+": "USR",  # replace any user tag with USR (the tag system is the same also in twitch)
            r"(.)\1\1+": r"\1\1",  # replace 3 consecutive chars with 2
            r"[\s]+": " ",  # remove consec spaces
            "#[a-z0-9]*": ""  # remove hashtags, they are not used in twitch chats
        }

        sbStem = SnowballStemmer("english", True)
        preprocess = ps.Preprocess(negations, emojis, regex_subs, sbStem)
        df = load_dataset(path, columns, final_columns)
        df = df[:500]
        df = preprocess.df_pre_process(df, "text", "sentiment")
        df = df['text']
        text = [remove_stopwords(x).translate(
            str.maketrans('', '', string.punctuation)
        ).translate(str.maketrans('', '', string.digits)) for x in df]
        ntf = Kmeanclustering.tfvec(df, text)

        return ntf, df


ntf, df = Kmeanclustering.main()


