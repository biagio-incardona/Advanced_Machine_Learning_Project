#!/usr/bin/env bash

while getopts t:y: flag
do
    case "${flag}" in 
        t) tc=${OPTARG};;
        y) yc=${OPTARG};;
    esac
done

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
    wget https://downloads.apache.org/kafka/2.6.3/kafka_2.12-2.6.3.tgz && \
    mv kafka_2.12-2.6.3.tgz kafka/setup;
fi

echo "starting ZooKeeper"

#docker stop kafkaZK >/dev/null

#docker container rm kafkaZK >/dev/null

docker run -e KAFKA_ACTION=start-zk --network tap --ip 10.0.100.22  -p 2181:2181 --name kafkaZK tap2:kafka >/dev/null &

echo "starting Kafka Server"

#docker stop kafkaServer >/dev/null

#docker container rm kafkaServer >/dev/null

docker run -e KAFKA_ACTION=start-kafka --network tap --ip 10.0.100.23  -p 9092:9092 --name kafkaServer tap2:kafka >/dev/null &

echo "starting LogStash"

#docker stop LogStash >/dev/null

#docker container rm LogStash >/dev/null

docker run --network tap --ip 10.0.100.11 --name LogStash tap2:logs >/dev/null &

echo "starting Python"

#docker stop Python >/dev/null

#docker container rm Python >/dev/null

docker run --network tap --ip 10.0.100.10 --name Python -e CHANNEL_TW=$tc -e CHANNEL_YT=$yc tap2:python >/dev/null &

echo "starting ElasticSearch"

#docker stop ElasticSearch >/dev/null

#docker container rm ElasticSearch >/dev/null

docker run -t  -p 9200:9200 -p 9300:9300 --ip 10.0.100.51 --name ElasticSearch --network tap -e "discovery.type=single-node"  tap2:es >/dev/null &

echo "starting kibana"

#docker stop Kibana >/dev/null

#docker container rm Kibana >/dev/null

docker run -p 5601:5601 --ip 10.0.100.52 --name Kibana --network tap tap2:kibana >/dev/null &

echo "wait a bit for starting spark"


echo "Waiting for Kibana to start WebServer..."

while ! nc -z 10.0.100.52 5601; do
  printf '.'
  sleep 1
done

echo "Kibana launched"
sleep 15

xdg-open http://localhost:5601/

sleep 10

echo "starting Spark"

spark/sparkSubmitPython.sh process.py "org.apache.spark:spark-streaming-kafka-0-8_2.11:2.4.5,org.elasticsearch:elasticsearch-hadoop:7.7.0" >/dev/null &

