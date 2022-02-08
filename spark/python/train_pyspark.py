# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 11:44:40 2022

@author: biagi
"""

from pyspark.sql.types import StructType, StructField, StringType, ArrayType, StringType
from pyspark.sql import SparkSession 
from pyspark.sql import functions as F
from pyspark.sql.functions import lower, when
from nltk.stem.snowball import SnowballStemmer
#from sparknlp.annotator import Stemmer
from pyspark.ml import Pipeline
from pyspark.ml.pipeline import Transformer
from pyspark.ml.feature import IDF, HashingTF, Tokenizer, StringIndexer
import nltk
nltk.download('stopwords')  

class TextPreprocessing(Transformer):
    def __init__(self, inputCol, outputCol='text'):
        self.inputCol = inputCol
        self.outputCol = outputCol
    def this():
        this(Identifiable.randomUID("textpreprocessing"))
    def copy(extra):
        defaultCopy(extra)
    def setDicts(self, negations, emojis):
        self.negations = negations
        self.emojis = emojis
    def _transform(self, sparkDF):
        sparkDF = sparkDF.withColumn("text", F.lower(sparkDF.text))
        print(self.inputCol[0])
        # replace emoticons
        for key in emojis.keys():
            sparkDF = sparkDF.withColumn(self.inputCol[0], F.regexp_replace(self.inputCol[0], key, emojis[key]))
        for key in negations.keys():
            sparkDF = sparkDF.withColumn(self.inputCol[0], F.regexp_replace(self.inputCol[0], key, negations[key]))
        sparkDF = sparkDF.withColumn(self.inputCol[0], F.regexp_replace(self.inputCol[0], r"https?://[^s]+", 'URL'))
        sparkDF = sparkDF.withColumn(self.inputCol[0], F.regexp_replace(self.inputCol[0], "www.[^ ]+", 'URL'))
        sparkDF = sparkDF.withColumn(self.inputCol[0], F.regexp_replace(self.inputCol[0], r"@[^\s]+", 'USR'))
        sparkDF = sparkDF.withColumn(self.inputCol[0], F.regexp_replace(self.inputCol[0], r"(.)\1\1+", r"\1\1"))
        sparkDF = sparkDF.withColumn(self.inputCol[0], F.regexp_replace(self.inputCol[0], r'(.)\1\1+', r'\.\.'))
        sparkDF = sparkDF.withColumn(self.inputCol[0], F.regexp_replace(self.inputCol[0], r'[\s]+', ' '))
        sparkDF = sparkDF.withColumn(self.inputCol[0], F.regexp_replace(self.inputCol[0], "#[a-z0-9]*", ''))
        return sparkDF
    
class SentimentPreprocessing(Transformer):
    def __init__(self, inputCol, outputCol='text'):
        self.inputCol = inputCol
        self.outputCol = outputCol
    def this():
        this(Identifiable.randomUID("sentimentpreprocessing"))
    def copy(extra):
        defaultCopy(extra)
    def _transform(self, sparkDF):
        sparkDF = sparkDF.withColumn(self.inputCol[0], when(df[self.inputCol[0]] > 1, 1).otherwise(0))
        return sparkDF
        
class StemmingPreprocessing(Transformer):
    def __init__(self, inputCol, outputCol='text'):
        self.inputCol = inputCol
        self.outputCol = outputCol
    def this():
        this(Identifiable.randomUID("stemmingpreprocessing"))
    def copy(extra):
        defaultCopy(extra)
    def setStemmer(self, stemmer):
    	self.sbStem = stemmer
    def udf_stemming(self, str, sbStem):
        stemmed_text = ""
        for word in str.split():
            word = self.sbStem.stem(word)
            stemmed_text += (word+" ")
        return stemmed_text
    def _transform(self, sparkDF):
        udf_f = F.udf(lambda x: udf_stemming(x, self.sbStem), StringType())
        sparkDF = sparkDF.withColumn(self.inputCol[0], udf_f(F.col(self.inputCol[0])))
        return sparkDF
    
def text_preprocessing(sparkDF, col, negations, emojis):
    sparkDF = sparkDF.withColumn(col, F.lower(sparkDF.text))
    # replace emoticons
    for key in emojis.keys():
        sparkDF = sparkDF.withColumn(col, F.regexp_replace(col, key, emojis[key]))
    for key in negations.keys():
        sparkDF = sparkDF.withColumn(col, F.regexp_replace(col, key, negations[key]))
    sparkDF = sparkDF.withColumn(col, F.regexp_replace(col, r"https?://[^s]+", 'URL'))
    sparkDF = sparkDF.withColumn(col, F.regexp_replace(col, "www.[^ ]+", 'URL'))
    sparkDF = sparkDF.withColumn(col, F.regexp_replace(col, r"@[^\s]+", 'USR'))
    sparkDF = sparkDF.withColumn(col, F.regexp_replace(col, r"(.)\1\1+", r"\1\1"))
    sparkDF = sparkDF.withColumn(col, F.regexp_replace(col, r'(.)\1\1+', r'\.\.'))
    sparkDF = sparkDF.withColumn(col, F.regexp_replace(col, r'[\s]+', ' '))
    sparkDF = sparkDF.withColumn(col, F.regexp_replace(col, "#[a-z0-9]*", ''))
    return sparkDF

def sentiment_preprocessing(sparkDF, col):
    sparkDF = sparkDF.withColumn(col, when(df[col] > 1, 1).otherwise(0))
    return sparkDF
    
def udf_stemming(str, sbStem):
	stemmed_text = ""
	for word in str.split():
		word = sbStem.stem(word)
		stemmed_text += (word+" ")
	return stemmed_text

def df_stemming(sparkDF, col):
	sbStem = SnowballStemmer("english", True)
	udf_f = F.udf(lambda x: udf_stemming(x, sbStem), StringType())
	sparkDF = sparkDF.withColumn(col, udf_f(F.col(col)))
	return sparkDF
	
def udf_tfidf(str, vectorizer):
	return vectorizer.transform(str)
	
def df_tfidf(sparkDF, vectorizer, col):
	udf_idf = F.udf(lambda x: udf_tfidf(x), StringType())
	sparkDF = sparkDF.withColumn(col, udf_idf(F.col(col), vectorizer))
	return sparkDF

DATASET_COLUMNS = StructType([
    StructField("sentiment", StringType(), True),
    StructField("ids", StringType(), True),
    StructField("date", StringType(), True),
    StructField("flag", StringType(), True),
    StructField("user", StringType(), True),
    StructField("text", StringType(), True)])

DATASET_ENCODING = "ISO-8859-1"
cols = ("ids", "date", "flag", "user")
spark = SparkSession.builder.master('local[*]').config("spark.driver.memory", "15g").appName('Tweets Sentiment').getOrCreate()
df = spark.read.csv("/opt/advm/dataset.csv",header = 'False',schema=DATASET_COLUMNS)
df = df.drop(*cols)
spark.sparkContext.setLogLevel('ERROR')

negations = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}

emojis = {":\)": 'smile', ":-\)": 'smile', ";d": 'wink', ":-E": 'vampire', ":\(": 'sad', 
          ':-\(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-\$': 'confused', 
          ':#': 'mute', ':X': 'mute', ':\^\)': 'smile', ':-&': 'confused', '\$_\$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O\.o': 'confused',
          '<\(-_-\)>': 'robot', 'd\[-_-]b': 'dj', ":'-\)": 'sadsmile', ';\)': 'wink', 
          ';-\)': 'wink', 'O:-\)': 'angel','O\*-\)': 'angel','\(:-D': 'gossip', '=\^.\^=': 'cat'}

#df = spark.read.csv("/opt/advm/dataset.csv",header = 'False',schema=DATASET_COLUMNS)

#import sparknlp
#from sparknlp.base import *
#from sparknlp.annotator import *

#stemmer = Stemmer().setInputCols(["text"])


#textProc = TextPreprocessing(["text"])
#textProc.setDicts(negations, emojis)
#sentProc = SentimentPreprocessing(["text"])
#stemProc = StemmingPreprocessing(["text"])
#stemProc.setStemmer(SnowballStemmer("english", True))

#pipeline = Pipeline().setStages([
#        textProc,
#        sentProc,
  #      stemProc
#    ]).fit(df).transform(df)

#pipeline.show()
#df2.show()
#waste=[print(" ") for i in range(1,100)]


df = sentiment_preprocessing(df, "sentiment")
(train_set, test_set) = df.randomSplit([0.95, 0.05], seed = 99)
#vectorizer = TfidfVectorizer(ngram_range = (1,2), max_features = 50000000)

tokenizer = Tokenizer(inputCol="text", outputCol = "words")
hashtf = HashingTF(numFeatures = 50000000, inputCol = "words", outputCol = "tf")
idf = IDF(inputCol='tf', outputCol='features')#.fit(train_set)
label_stringIDx = StringIndexer(inputCol="sentiment", outputCol = "label")
pipeline = Pipeline(stages = [tokenizer, hashtf, idf, label_stringIDx]).fit(train_set)
train_set = pipeline.transform(train_set)
test_set = pipeline.transform(test_set)
train_set.show()
test_set.show()
df.show()
