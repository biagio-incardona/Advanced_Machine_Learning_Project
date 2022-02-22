from sklearn.model_selection import train_test_split
from importlib.machinery import SourceFileLoader
from nltk.stem import SnowballStemmer

import pickle
# save the classifier


# importing the add module from the custom packases using the path
# useful only if you are running in IPython
foo = SourceFileLoader(
    "settings", "C:\\Users\\biagi\\Desktop\\university\\Second Year\\First Semester\\advanced machine learning\\prog\\Advanced_Machine_Learning_Project\\settings.py"
 ).load_module()
from settings import BASE_DIR
foo = SourceFileLoader("model_selection", "{BaseDir}/Advanced_Machine_Learning_Project/ML/model_selection.py".format(BaseDir=BASE_DIR) )
foo = SourceFileLoader(
    "TFIDF_Models", "{BaseDir}/Advanced_Machine_Learning_Project/ML/TFIDF_Models.py".format(BaseDir=BASE_DIR)
).load_module()
foo = SourceFileLoader(
    "Preprocess", "{BaseDir}/Advanced_Machine_Learning_Project/ML/Preprocess.py".format(BaseDir=BASE_DIR)
).load_module()

import TFIDF_Models #import TFIDFLogisticRegression
import Preprocess as ps
import model_selection as ms
import random
random.seed(16)

path = "C:/Users/biagi/Desktop/university/Second Year/First Semester/advanced machine learning/data/dataset.csv".format(BaseDir=BASE_DIR)
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
df = ms.load_dataset(path, columns, final_columns)
df = ms.resize(df, 500, "sentiment", 4)
df = preprocess.df_pre_process(df, "text", "sentiment")
X_train, X_test, Y_train, Y_test = ms.df_train_test_split(df, "text", "sentiment", test_size=0.05)
model_json = {
    "model": "TFIDFLogisticRegression",
    "model__C": 0.5854128924676655,
    "model__fit_intercept": False,
    "model__l1_ratio": 0.18870192595698554,
    "model__max_iter": 495,
    "model__ngram_range": [
        1,
        2
    ],
    "model__penalty": "elasticnet",
    "model__solver": "saga",
    "model__tfidf_max_features": 14227465,
    "model__tol": 0.00011935453680483443,
    "mean_test_score": 0.8188717105263159,
    "mean_score_time": 10.590722703933716,
    "mean_fit_time": 18724.629264974596,
    "mean_train_score": 0.8325748355263158
}
model = TFIDF_Models.TFIDFLogisticRegression(
    tfidf_max_features=model_json["model__tfidf_max_features"],
    ngram_range=(model_json["model__ngram_range"][0], model_json["model__ngram_range"][1]),
    penalty=model_json["model__penalty"],
    tol=model_json["model__tol"],
    C=model_json["model__C"],
    fit_intercept=model_json["model__fit_intercept"],
    solver=model_json["model__solver"],
    max_iter=model_json["model__max_iter"]+100,
    verbose=1,
    n_jobs = -1,
    l1_ratio=model_json["model__l1_ratio"]
)

#model.fit(X_train, Y_train)
#print(model)
#with open('TFIDF_logisticRegression.pkl', 'wb') as fid:
#    pickle.dump(model, fid)

with open('C:/Users/biagi/Desktop/TFIDF_logisticRegression.pkl', 'rb') as fid:
    model = pickle.load(fid)

ms.model_evaluate(model, X_test, Y_test)

print(model.predict_proba(X_test))