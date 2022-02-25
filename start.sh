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

docker run -e KAFKA_ACTION=start-zk --network advm --ip 10.0.100.22  -p 2181:2181 --name kafkaZK advm:kafka >/dev/null &

echo "starting Kafka Server"

#docker stop kafkaServer >/dev/null

#docker container rm kafkaServer >/dev/null

docker run -e KAFKA_ACTION=start-kafka --network advm --ip 10.0.100.23  -p 9092:9092 --name kafkaServer advm:kafka >/dev/null &

echo "starting LogStash"

#docker stop LogStash >/dev/null

#docker container rm LogStash >/dev/null

docker run --network advm --ip 10.0.100.11 --name LogStash advm:logs >/dev/null &

echo "starting Python"

#docker stop Python >/dev/null

#docker container rm Python >/dev/null

docker run --network advm --ip 10.0.100.10 --name Python -e CHANNEL_TW=$tc advm:python >/dev/null &

echo "starting ElasticSearch"

#docker stop ElasticSearch >/dev/null

#docker container rm ElasticSearch >/dev/null

docker run -t  -p 9200:9200 -p 9300:9300 --ip 10.0.100.51 --name ElasticSearch --network advm -e "discovery.type=single-node"  advm:es >/dev/null &

echo "starting kibana"

#docker stop Kibana >/dev/null

#docker container rm Kibana >/dev/null

docker run -p 5601:5601 --ip 10.0.100.52 --name Kibana --network advm advm:kibana >/dev/null &

echo "wait a bit for starting spark"


echo "Waiting for Kibana to start WebServer..."

while ! nc -z 10.0.100.52 5601; do
  printf '.'
  sleep 1
done

sleep 15
echo "Kibana launched"
xdg-open http://localhost:5601/ &

sleep 10
echo "Loading dashboard"
curl -XPOST 10.0.100.52:5601/api/kibana/dashboards/import -H 'kbn-xsrf:true' -H 'Content-type:application/json' -d @./kibana/dashboard.json

echo "starting Spark"
docker run -e SPARK_ACTION=spark-submit-python -p 4040:4040 -it --network advm --name sparkSubmit advm:spark process.py "org.apache.spark:spark-streaming-kafka-0-8_2.11:2.4.5,org.elasticsearch:elasticsearch-hadoop:7.7.0"

