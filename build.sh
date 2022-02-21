#!/usr/bin/env bash

SPARK_DIR="./spark/setup"
KAFKA_DIR="./kafka/setup"

rm spark/setup/placeholder.txt 2>&1 >/dev/null
rm kafka/setup/placeholder.txt 2>&1 >/dev/null

echo "Checking dependencies..."
if [ "$(ls -A $SPARK_DIR)" ]; then
     echo "Spark's dependencies found..."
else
    echo "Installing Spark's dependencies..."
    wget https://downloads.apache.org/spark/spark-2.4.8/spark-2.4.8-bin-hadoop2.7.tgz && \
    mv spark-2.4.8-bin-hadoop2.7.tgz spark/setup;
fi
if [ "$(ls -A $KAFKA_DIR)" ]; then
     echo "Kafka's dependencies found..."
else
    echo "Installing Kafka's dependencies..."
    wget https://downloads.apache.org/kafka/3.0.0/kafka_2.12-3.0.0.tgz && \
    mv kafka_2.12-3.0.0.tgz kafka/setup;
fi


docker build spark/. --tag advm:spark |& tee ./build_spark_log.txt

echo "--------------------------------------------"

docker build elasticsearch/. --tag advm:es |& tee ./build_es_log.txt

echo "--------------------------------------------"

docker build python/. --tag advm:python |& tee ./build_python_log.txt

echo "--------------------------------------------"

docker build logstash/. --tag advm:logs |& tee ./build_logstash_log.txt

echo "--------------------------------------------"

docker build kafka/. --tag advm:kafka |& tee ./build_kafka_log.txt

echo "--------------------------------------------"

docker build kibana/. --tag advm:kibana |& tee ./build_kibana_log.txt
