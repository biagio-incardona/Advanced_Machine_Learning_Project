FROM openjdk:8-jre

ENV PATH $SPARK_DIR/bin:$PATH
ENV SPARK_VERSION=2.4.8
ENV SPARK_DIR=/opt/spark
ENV PATH $SPARK_DIR/bin:$PATH
ENV PYSPARK_PYTHON=python3.6

ADD setup/spark-${SPARK_VERSION}-bin-hadoop2.7.tgz /opt


RUN apt-get update && apt-get -y install bash
RUN apt -y install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev wget libbz2-dev
RUN wget https://www.python.org/ftp/python/3.6.9/Python-3.6.9.tgz && \
	tar xzf Python-3.6.9.tgz && \
	cd Python-3.6.9 && \
	./configure && \
	make altinstall


RUN python3.6 -V

COPY get-pip.py .
RUN python3.6 get-pip.py

RUN python3.6 -m pip install pyspark==3.2.1 elasticsearch==7.7.0 kafka-python==2.0.2 vaderSentiment spaCy
RUN python3.6 -m pip install numpy==1.19.5
RUN python3.6 -m spacy download en_core_web_sm
RUN python3.6 -m pip install scikit-learn==0.24.0
RUN python3.6 -m pip install nltk==3.6.7
# Create Sym Link 
RUN ln -s /opt/spark-${SPARK_VERSION}-bin-hadoop2.7 ${SPARK_DIR} 

ADD python/*  /opt/advm/

RUN unzip ./opt/advm/TFIDF_logisticRegression.zip -d ./opt/advm

# Add Spark Manager
ADD spark-manager.sh $SPARK_DIR/bin/spark-manager

WORKDIR ${SPARK_DIR}
ENTRYPOINT [ "bin/spark-manager" ]

