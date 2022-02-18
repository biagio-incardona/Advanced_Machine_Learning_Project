import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import random
import json
from importlib.machinery import SourceFileLoader

# importing the add module from the custom packases using the path
# useful only if you are running in IPython
#foo = SourceFileLoader(
#    "settings", "C:\\Users\\biagi\\Desktop\\university\\Second Year\\First Semester\\advanced machine learning\\prog\\Advanced_Machine_Learning_Project\\settings.py"
#).load_module()

from settings import BASE_DIR
foo = SourceFileLoader(
    "TFIDF_Models", "{BaseDir}/Advanced_Machine_Learning_Project/ML/TFIDF_Models.py".format(BaseDir=BASE_DIR)
).load_module()
foo = SourceFileLoader(
    "w2v_Models", "{BaseDir}/Advanced_Machine_Learning_Project/ML/w2v_Models.py".format(BaseDir=BASE_DIR)
).load_module()
foo = SourceFileLoader(
    "Preprocess", "{BaseDir}/Advanced_Machine_Learning_Project/ML/Preprocess.py".format(BaseDir=BASE_DIR)
).load_module()
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
import scipy.stats.distributions as dists
import TFIDF_Models as cnb
import w2v_Models as w2v
import Preprocess as ps

# nltk.download('wordnet')
# nltk.download('stopwords')
random.seed(16)


def load_dataset(path, columns, final_columns):
    print("loading dataset...")
    df = pd.read_csv(path, encoding='ISO-8859-1', names=columns)
    df = df[final_columns]
    print("...dataset loaded")
    return df


def resize(dataset, n_obs, var_check, value):
    import math
    half = math.floor(n_obs / 2)
    data_pos = dataset[dataset[var_check] == value]
    data_neg = dataset[dataset[var_check] != value]
    data_pos = data_pos.iloc[random.sample(range(0, data_pos.count()[0] + 1), half)]
    data_neg = data_neg.iloc[random.sample(range(0, data_neg.count()[0] + 1), half)]
    dataset = pd.concat([data_pos, data_neg])
    dataset = dataset.sample(frac=1)
    dataset = dataset.reset_index(drop=True)
    return dataset


def df_train_test_split(df, var_text, var_sentiment, test_size):
    print("starting splitting dataset...")
    X = df[var_text]
    Y = df[var_sentiment]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
    print("...dataset splitted")
    return (X_train, X_test, Y_train, Y_test)


def model_evaluate(model, X, Y):
    Y_pred = model.predict(X)
    print(classification_report(Y, Y_pred))
    cf_matrix = confusion_matrix(Y, Y_pred)
    categories = ['Negative', 'Positive']
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_percentages = ['{}'.format(value) for value in cf_matrix.flatten()]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='',
                xticklabels=categories, yticklabels=categories)
    plt.xlabel("Predicted values", fontdict={'size': 14}, labelpad=10)
    plt.ylabel("Actual values", fontdict={'size': 14}, labelpad=10)
    plt.title("Confusion Matrix", fontdict={'size': 18}, pad=20)


def save_results(search, file_name):
    search = search.cv_results_
    for i in range(len(search['params'])):
        model = search['params'][i]
        model['model'] = str(model['model'])
        model['model'] = model['model'][:model['model'].find('(')]
        model['mean_test_score'] = search['mean_test_score'][i]
        model['mean_score_time'] = search['mean_score_time'][i]
        model['mean_fit_time'] = search['mean_fit_time'][i]
        model["mean_train_score"] = search['mean_train_score'][i]
        with open("{BaseDir}/Advanced_Machine_Learning_Project/ML/{FileName}.json".format(BaseDir=BASE_DIR,
                                                                                          FileName=file_name),
                  'a') as f:
            json.dump(model, f, indent=4)


# must be removed when sending the project
def apply_random_search(pip, params_grid, X_train, Y_train, results_file_name):
    results = pd.DataFrame(columns=["model", "mean_fit_time", "mean_score_time", "mean_train_score", "mean_test_score"])
    #for i in range(10):
    trys = RandomizedSearchCV(pip, param_distributions=params_grid, n_iter=15, n_jobs=-1, return_train_score=True)
    search = trys.fit(X_train, Y_train)
    save_results(search, results_file_name)

def first_screening(df):
    X_train, X_test, Y_train, Y_test = df_train_test_split(df, "text", "sentiment", test_size=0.05)

    w2v1 = w2v.W2VDecisionTreeClassifier()

    pip = Pipeline([('model', w2v1)])
    params_grid = [dict(model=[w2v.W2VDecisionTreeClassifier()],
                        model__vector_size=dists.randint(1, 10000 + 1),
                        model__window=dists.randint(1, 10 + 1),
                        model__min_count=dists.randint(1, 10 + 1),
                        model__epochs=dists.randint(5, 25 + 1),
                        model__criterion=['gini', 'entropy'],
                        model__splitter=['best', 'random'],
                        model__max_depth=dists.randint(-2, 10 + 1),
                        model__min_samples_split=dists.randint(1, 20 + 1),
                        model__min_samples_leaf=dists.randint(1, 20 + 1),
                        model__min_weight_fraction_leaf=dists.uniform(0, 0.5),
                        model__max_features=[None, "auto", "sqrt", "log2"],
                        model__max_leaf_nodes=dists.randint(-50, 1000 + 1),
                        model__min_impurity_decrease=dists.uniform(0.0, 10.0),
                        model__ccp_alpha=dists.uniform(0.0, 10.0)
                        ),
                   dict(model=[w2v.W2VGaussianNB()],
                        model__vector_size=dists.randint(1, 10000 + 1),
                        model__window=dists.randint(1, 10 + 1),
                        model__min_count=dists.randint(1, 10 + 1),
                        model__epochs=dists.randint(5, 25 + 1)
                        ),
                   dict(model=[w2v.W2VLinearSVC()],
                        model__vector_size=dists.randint(1, 10000 + 1),
                        model__window=dists.randint(1, 10 + 1),
                        model__min_count=dists.randint(1, 10 + 1),
                        model__epochs=dists.randint(5, 25 + 1),
                        model__penalty=['l1', 'l2'],
                        model__dual=[False],
                        model__tol=dists.uniform(0.00001, 0.001),
                        model__C=dists.uniform(0.1, 2.0),
                        model__intercept_scaling=dists.uniform(0.1, 10),
                        model__max_iter=dists.randint(100, 2000 + 1)
                        ),
                   dict(model=[w2v.W2VMultinomialNB()],
                        model__vector_size=dists.randint(1, 10000 + 1),
                        model__window=dists.randint(1, 10 + 1),
                        model__min_count=dists.randint(1, 10 + 1),
                        model__epochs=dists.randint(5, 25 + 1),
                        model__alpha=dists.uniform(0, 5.0)
                        ),
                   dict(model=[w2v.W2VLogisticRegression()],
                        model__vector_size=dists.randint(1, 10000 + 1),
                        model__window=dists.randint(1, 10 + 1),
                        model__min_count=dists.randint(1, 10 + 1),
                        model__epochs=dists.randint(5, 25 + 1),
                        model__penalty=['l1', 'l2', 'elasticnet', 'none'],
                        model__tol=dists.uniform(0.00001, 0.001),
                        model__C=dists.uniform(0.1, 2.0),
                        model__fit_intercept=[True, False],
                        model__solver=['saga'],
                        model__max_iter=dists.randint(100, 1000 + 1),
                        model__l1_ratio=dists.uniform(0.0, 1.0)
                        ),
                   dict(model=[cnb.TFIDFDecisionTreeClassifier()],
                        model__tfidf_max_features=dists.randint(1, 50000000),
                        model__ngram_range=[(1, 2)],
                        model__criterion=['gini', 'entropy'],
                        model__splitter=['best', 'random'],
                        model__max_depth=dists.randint(-2, 10 + 1),
                        model__min_samples_split=dists.randint(2, 20 + 1),
                        model__min_samples_leaf=dists.randint(1, 20 + 1),
                        model__min_weight_fraction_leaf=dists.uniform(0, 0.5),
                        model__max_features=[None, "auto", "sqrt", "log2"],
                        model__max_leaf_nodes=dists.randint(-50, 1000 + 1),
                        model__min_impurity_decrease=dists.uniform(0.0, 10.0),
                        model__ccp_alpha=dists.uniform(0.0, 10.0)
                        ),
                   dict(model=[cnb.TFIDFGaussianNB()],
                        model__tfidf_max_features=dists.randint(1, 50000000),
                        model__ngram_range=[(1, 2)]
                        ),
                   dict(model=[cnb.TFIDFLinearSVC()],
                        model__tfidf_max_features=dists.randint(1, 50000000),
                        model__ngram_range=[(1, 2)],
                        model__penalty=['l1', 'l2'],
                        model__dual=[False],
                        model__tol=dists.uniform(0.00001, 0.001),
                        model__C=dists.uniform(0.1, 2.0),
                        model__intercept_scaling=dists.uniform(0.1, 10),
                        model__max_iter=dists.randint(100, 2000 + 1)
                        ),
                   dict(model=[cnb.TFIDFMultinomialNB()],
                        model__tfidf_max_features=dists.randint(1, 50000000),
                        model__ngram_range=[(1, 2)],
                        model__alpha=dists.uniform(0, 5.0)
                        ),
                   dict(model=[cnb.TFIDFLogisticRegression()],
                        model__tfidf_max_features=dists.randint(1, 50000000),
                        model__ngram_range=[(1, 2)],
                        model__penalty=['l1', 'l2', 'elasticnet', 'none'],
                        model__tol=dists.uniform(0.00001, 0.001),
                        model__C=dists.uniform(0.1, 2.0),
                        model__fit_intercept=[True, False],
                        model__solver=['saga'],
                        model__max_iter=dists.randint(100, 1000 + 1),
                        model__l1_ratio=dists.uniform(0.0, 1.0)
                        )]

    apply_random_search(pip, params_grid, X_train, Y_train, "RandomSearchModelResults")

def second_screening(df):
    X_train, X_test, Y_train, Y_test = df_train_test_split(df, "text", "sentiment", test_size=0.05)

    w2v1 = w2v.W2VDecisionTreeClassifier()

    pip = Pipeline([('model', w2v1)])
    params_grid = [dict(model=[cnb.TFIDFMultinomialNB()],
                        model__tfidf_max_features=dists.randint(1, 50000000),
                        model__ngram_range=[(1, 2)],
                        model__alpha=dists.uniform(0, 5.0)
                        ),
                   dict(model=[cnb.TFIDFLogisticRegression()],
                        model__tfidf_max_features=dists.randint(1, 50000000),
                        model__ngram_range=[(1, 2)],
                        model__penalty=['l1', 'l2', 'elasticnet', 'none'],
                        model__tol=dists.uniform(0.00001, 0.001),
                        model__C=dists.uniform(0.1, 2.0),
                        model__fit_intercept=[True, False],
                        model__solver=['saga'],
                        model__max_iter=dists.randint(100, 1000 + 1),
                        model__l1_ratio=dists.uniform(0.0, 1.0)
                    )]

    apply_random_search(pip, params_grid, X_train, Y_train, "second_screening_results")


def main():
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
    # df = resize(df, 50000, "sentiment", 4)
    df = preprocess.df_pre_process(df, "text", "sentiment")
    #first_screening(df)
    second_screening(df)

if __name__ == "__main__":
    main()
