#!/usr/bin/env bash

echo "Stopping containers..."

docker stop kafkaZK && \
docker stop kafkaServer && \
docker stop LogStash && \
docker stop ElasticSearch && \
docker stop Kibana && \
docker stop sparkSubmit && \
docker stop Python && \


echo "Deleting containers..." && \

docker container rm kafkaZK && \
docker container rm kafkaServer && \
docker container rm LogStash && \
docker container rm ElasticSearch && \
docker container rm Kibana && \
docker container rm sparkSubmit && \
docker container rm Python
