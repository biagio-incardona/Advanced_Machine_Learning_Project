from __future__ import print_function

import numpy
import sys
import json
from pyspark import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.streaming import StreamingContext
import pyspark
from datetime import datetime
import time
from pyspark import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql.session import SparkSession
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import pyspark.sql.types as tp
from pyspark.ml import Pipeline
from pyspark.sql import Row
import os
from collections import Counter
from string import punctuation
import TFIDF_Models as models
import nltk
import STClustering
from nltk.stem import SnowballStemmer
import Preprocess as ps
import pickle

nltk.download('stopwords')
model = None
with open('/opt/advm/TFIDF_logisticRegression.pkl', 'rb') as file:
    model = pickle.load(file)
    
clustering_model = STClustering.STClustering(r=0.75, gap_time = 5000)

get_twitch_schema = tp.StructType([
    tp.StructField(name = 'username', dataType = tp.StringType(),  nullable = True),
    tp.StructField(name = 'timestamp', dataType = tp.LongType(),  nullable = True),
    tp.StructField(name = 'mex', dataType = tp.StringType(),  nullable = True),
    tp.StructField(name = 'engagement', dataType = tp.FloatType(), nullable = True),
    tp.StructField(name = 'source', dataType = tp.StringType(),  nullable = True)
])

def get_sentiment(text):
    value = model.predict_proba([text])
    value = value[0][1]
    print(value)
    return value

def process(key, rdd):
    global spark
    print(key)
    print(rdd)
    twitch_chat = rdd.map(lambda value: json.loads(value[1])).map(
        lambda json_object:(
            json.loads(json_object["message"].encode("ascii", "ignore"))["message"],
            json.loads(json_object["message"].encode("ascii", "ignore"))["username"],
            float(json.loads(json_object["message"].encode("ascii", "ignore"))["engagement"]),
            int(json.loads(json_object["message"].encode("ascii", "ignore"))["timestamp"]),
            json.loads(json_object["message"].encode("ascii", "ignore"))["source"]
        )
    )

    twitch_message = twitch_chat.collect()
    if not twitch_message:
        print("No Messages")
        return
    mex = twitch_message[0][0]
    time = twitch_message[0][3]
    print(mex)
    mex2 = mex
    mex = mex.encode("ascii", "ignore")
    mex2 = preprocessor.text_preprocess(mex2)
    sentiment = get_sentiment(mex2)

    rowRdd = twitch_chat.map(lambda t:
        Row(
            mex = t[0], username = t[1],
            engagement = t[2], timestamp = t[3],
            source = t[4]
        )
    )

    dataFrame = spark.createDataFrame(rowRdd, schema = get_twitch_schema)    
    
    new = dataFrame.rdd.map(lambda x:
        {
            'username' : x['username'],
            'timestamp' : x['timestamp'],
            'mex' : x['mex'],
            'engagement' : x['engagement'],
            'source' : x['source'],
            'mex_sentiment' : sentiment
        }
    )

    final_rdd = new.map(json.dumps).map(lambda x: ('key', x))
    final_rdd.saveAsNewAPIHadoopFile(
        path='-',
        outputFormatClass="org.elasticsearch.hadoop.mr.EsOutputFormat",
        keyClass="org.apache.hadoop.io.NullWritable",
        valueClass="org.elasticsearch.hadoop.mr.LinkedMapWritable",
        conf=es_write_conf
    )
    time_gap_reached = clustering_model.insert(mex2, time)
    if time_gap_reached == True:
    	clusters = clustering_model.get_clusters()
    	clust_sparkDf = spark.createDataFrame(clusters)
    	clust_rdd = clust_sparkDf.rdd.map(lambda x:
    		{
    			'keyword' : x['cluster'],
    			'weight' : x['weight'],
    			'timestamp' : time
    		}
    	)
    	clust_final_rdd = clust_rdd.map(json.dumps).map(lambda x: ('key',x))
    	clust_final_rdd.saveAsNewAPIHadoopFile(
    		path='-',
    		outputFormatClass="org.elasticsearch.hadoop.mr.EsOutputFormat",
        	keyClass="org.apache.hadoop.io.NullWritable",
        	valueClass="org.elasticsearch.hadoop.mr.LinkedMapWritable",
        	conf=es_write_clust
    	)
    
    
    
negations = {"isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
                 "haven't": "have not", "hasn't": "has not", "hadn't": "had not", "won't": "will not",
                 "wouldn't": "would not", "don't": "do not", "doesn't": "does not", "didn't": "did not",
                 "can't": "can not", "couldn't": "could not", "shouldn't": "should not", "mightn't": "might not",
                 "mustn't": "must not"}

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
preprocessor = ps.Preprocess(negations, emojis, regex_subs, sbStem)

sc = SparkContext(appName="SparkSentimentTopicES")
spark = SparkSession(sc)
sc.setLogLevel("WARN")
ssc = StreamingContext(sc, 1)

conf = SparkConf(loadDefaults=False)
conf.set("es.index.auto.create", "true")

brokers="10.0.100.23:9092"
topic = "data"

kvs = KafkaUtils.createDirectStream(ssc, [topic], {"metadata.broker.list": brokers})
kvs.foreachRDD(process)

mapping = {
    "mappings": {
        "properties": {
            "timestamp": {
                "type": "date"
            }
        }
    }
}

from elasticsearch import Elasticsearch
elastic_host="10.0.100.51"
elastic_index="data"
elastic_index_clustering = "clustering"
elastic_document="_doc"
elastic_port = '9200'
elastic_is_json = "yes"

es_write_conf = {
    "es.nodes" : elastic_host,
    "es.port" : elastic_port,
    "es.resource" : '%s/%s' % (elastic_index,elastic_document),
    "es.input.json" : elastic_is_json,
 #   "mapred.reduce.tasks.speculative.execution": "false", #test
 #   "mapred.map.tasks.speculative.execution": "false", #test

}

es_write_clust = {
	"es.nodes" : elastic_host,
	"es.port" : elastic_port,
	"es.resource" : '%s/%s' % (elastic_index_clustering, elastic_document),
	"es.input.json" : elastic_is_json,
}

host = {
	'host' : elastic_host,
	'port' : elastic_port
}

#elastic = Elasticsearch(hosts=[host,])#elastic_host])
elastic = Elasticsearch(hosts=[host,],node_class="requests")

response = elastic.indices.create(
    index=elastic_index,
    body=mapping,
    ignore=400 # ignore 400 already exists code
)

cluster_response = elastic.indices.create(
	index=elastic_index_clustering,
	body=mapping,
	ignore=400
)

if 'acknowledged' in response:
    if response['acknowledged'] == True:
        print ("INDEX MAPPING SUCCESS FOR INDEX:", response['index'])

if 'acknowledged' in cluster_response:
    if cluster_response['acknowledged'] == True:
        print ("INDEX MAPPING SUCCESS FOR INDEX:", cluster_response['index'])

# catch API error response
elif 'error' in response:
    print ("ERROR:", response['error']['root_cause'])
    print ("TYPE:", response['error']['type'])



ssc.start()
ssc.awaitTermination()
