FROM openjdk:8-jre

ENV PATH $SPARK_DIR/bin:$PATH
ENV SPARK_VERSION=2.4.8
#ENV SPARK_VERSION=3.2.0
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

#COPY pip3.4 /bin
	
#RUN apt -y install python3.7
#RUN apt-get -y install python3.7.8
#RUN apt-get -y install python3-pip
#RUN python3.4 -m pip install pip
#COPY pip-19.1-py2.py3-none-any.whl .
#RUN python3.4 pip-19.1-py2.py3-none-any.whl/pip install --no-index pip-19.1-py2.py3-none-any.whl

#RUN python3.4 -m ensurepip
COPY get-pip.py .
#RUN curl https://bootstrap.pypa.io/pip/2.7/get-pip.py -o get-pip.py
RUN python3.6 get-pip.py

RUN python3.6 -m pip install pyspark elasticsearch kafka-python vaderSentiment spaCy
RUN python3.6 -m pip install numpy
#RUN pip install -U pip setuptools wheel
#RUN git clone https://github.com/explosion/spaCy
#RUN wget https://files.pythonhosted.org/packages/6b/a2/61cb20220e57eb9f53bba1f5de29da1ed12020af597c3a9a2229e43e75d7/spacy-legacy-3.0.8.tar.gz
#RUN pip install spacy-legacy-3.0.8.tar.gz
#RUN cd spaCy && pip install -r requirements.txt && python setup.py build_ext --inplace && pip install .
RUN python3.6 -m spacy download en_core_web_sm
#RUN pip install spacy==2.3.2


#medium version ~ 100 MB
#RUN python3 -m spacy download en_core_web_md
#RUN python -m spacy download en_core_web_trf

#large version ~ 800 MB, I've not enough GBs :(
#RUN python -m spacy download en_core_web_lg


# Create Sym Link 
RUN ln -s /opt/spark-${SPARK_VERSION}-bin-hadoop2.7 ${SPARK_DIR} 

ADD python/*  /opt/advm/

# Add Spark Manager
ADD spark-manager.sh $SPARK_DIR/bin/spark-manager

WORKDIR ${SPARK_DIR}
ENTRYPOINT [ "bin/spark-manager" ]

