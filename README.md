# Advanced_Machine_Learning_Project


## Requirements

Docker and curl are required for correct running, the rest of dependencies and necessary files will be automatically added at first start.

## Guide

***First, move into the project directory***

#### 1. First start (Or if you had deleted the docker images)
If this is the first time you are running the project (or your docker images have been deleted) run the following script, then go to step 2.

```shell

$ ./build.sh

```
NOTE: This step may take a time depending on the available internet connection

#### 2. Start

To run the project you just need to run this script 

```shell

$ ./start.sh -t "lowcase, Twitch channel name"

```

#### Stop

To stop the project you just need to run this script. All containers will be stoppend and deleted 

```shell

$ ./stop.sh

```
