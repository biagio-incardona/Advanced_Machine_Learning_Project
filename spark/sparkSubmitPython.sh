#!/usr/bin/env bash
# Stop
#docker stop sparkSubmit

# Remove previuos container 
#docker container rm sparkSubmit

docker run -e SPARK_ACTION=spark-submit-python -p 4040:4040 -it --network advm_net --name sparkSubmit advm:spark $1 $2
