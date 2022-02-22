#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# In[2]:


def load_dataset(path, columns):
    print("loading dataset...")
    df = pd.read_csv(path, encoding='ISO-8859-1', names=columns)
    print("...dataset loaded")
    return df
path = ("C:\\Users\\USER\\OneDrive\\Desktop\ML\senti.csv")
columns = ["sentiment", "ids", "date", "flag", "user", "text"]


df = load_dataset(path, columns)


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


#the shape shows us the number of values in the given dataset and the number of columns respectively.


# In[7]:


df.describe()


# In[9]:


df.dtypes


# In[10]:


df.nunique()


# In[11]:


#shows the unique set of values 


# In[12]:


df = pd.concat([df.query("sentiment==0"), df.query("sentiment==4")])
df.sentiment = df.sentiment.map({0:0, 4:1})
df =  shuffle(df).reset_index(drop=True)


# In[13]:


df.isnull().sum()


# In[14]:


#here we have checked for the null values 


# In[15]:


#we find which all users had the most tweets and plot it


# In[16]:


users = df['user'].value_counts()[:10]
users.plot(kind='bar', color='purple')


# In[17]:


#class distribution


# In[153]:


def word_count(sentence):
    return len(sentence.split())
    
df['word count'] = df['text'].apply(word_count)
df.head(3)


# In[ ]:


#We can see the word count per each user from the above assessment.


# In[186]:


# plot word count distribution for both positive and negative sentiments
x = df['word count'][df.sentiment == 1]
y = df['word count'][df.sentiment == 0]
plt.figure(figsize=(12,6))
plt.xlim(0,45)
plt.xlabel('word count')
plt.ylabel('frequency')
g = plt.hist([x, y], color=['r','b'], alpha=0.5, label=['positive','negative'])
plt.legend(loc='upper right')


# In[ ]:


#From the graph above,we can analyse that most sentences fall between 5–10 words.


# In[ ]:


#Commonly occuring words and stop words


# In[20]:


from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop=set(stopwords.words('english'))


# In[21]:


#here we are going to find the most commonly occuring words in our dataset


# In[22]:


corpus = []

word = df['text'].str.split()
new = word.values.tolist()
corpus=[word for i in new for word in i]


# In[23]:


from collections import Counter

counter=Counter(corpus)
most=counter.most_common(100)

x, y= [], []
for word,count in most[:50]:

    if word not in stop:
        x.append(word)
        y.append(count)
        
plt.bar(x,y, color='red')


# In[24]:


pip install plotly


# In[25]:


import plotly.express as px

temp = pd.DataFrame(most)


fig = px.treemap(temp, path=[0], values=1,title='Tree of words most tweeted')
fig.show()


# In[26]:


#here we can see the most commonly used words for tweets in both bar graph and as tree form.


# In[27]:


#now we check for the commonly occuring stop words


# In[28]:


from collections import defaultdict

dic=defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1

top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 
x,y=zip(*top)
plt.bar(x,y, color='green')


# In[29]:


temp = pd.DataFrame(top)
fig = px.treemap(temp, path=[0], values=1,title='Tree of words most tweeted in stop word list')
fig.show()


# In[30]:


#we have successfully plotted the bar graph and the tree for most commonly occuring stop words


# In[31]:


#length of tweets


# In[32]:


df['text'].str.len().hist()


# In[33]:


min(df['text'].str.len())


# In[34]:


max(df['text'].str.len())


# In[ ]:


#N-grams


# In[36]:


from sklearn.feature_extraction.text import CountVectorizer

def get_top_word_bigrams(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def get_top_word_trigrams(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# In[37]:


plt.figure(figsize=(10,5))
top_tweet_bigrams=get_top_word_bigrams(df['text'])[:20]
x,y=map(list,zip(*top_tweet_bigrams))
sns.barplot(x=y,y=x)


# In[97]:


plt.figure(figsize=(10,5))
top_tweet_trigrams=get_top_word_trigrams(df['text'])[:20]
x,y=map(list,zip(*top_tweet_trigrams))
sns.barplot(x=y,y=x)


# In[ ]:


#In these analysis we are using n-grams,In practice, n-gram models have been shown to be extremely effective in modeling language data. The above two models with contiguous sequence of n items in our dataset. The first plot is  bigram which is of size 2 , hence they will be considering sequence of 2 items. The second one is  bigram which is of size 2 , hence they will be considering sequence of 2 items. 


# In[38]:


#Topic modelling with LDA 


# In[ ]:


#Topic modeling is a type of statistical modeling for discovering the importnt topics that occur in a collection of documents. Latent Dirichlet Allocation (LDA) is an example of topic model and is used to classify text in a document. Before getting into topic modeling we have to pre-process our data.Topic models can help to organize and offer insights for us to understand large collections of unstructured text bodies.


# In[ ]:


#As for preprocessing our data we have to follow some steps.
#We have to tokenize which is the process by which sentences are converted to a list of tokens or words.
#We have to remove the stop words in our data.
#We have to lemmatize our data which is to reduce the inflectional forms of each word into a common base or root.


# In[39]:


from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


# In[40]:


copy_df = df.copy()
#here we are only taking a fraction of dataset because lemmatization demands much time for processing
df = df.sample(frac=0.1)


# In[43]:


import nltk
nltk.download('punkt')
nltk.download('wordnet')


# In[44]:


def preprocess_corpus(df):
    corpus = []
    stem=PorterStemmer()
    lem=WordNetLemmatizer()
    
    for tweet in df['text']:
        tweets = [t for t in word_tokenize(tweet) if (t not in stop)]
        tweets = [lem.lemmatize(t) for t in tweets if len(t)>2]
        corpus.append(tweets)
        
    return corpus

corpus = preprocess_corpus(df)


# In[47]:


pip install gensim


# In[48]:


from gensim import corpora
import gensim

dic=gensim.corpora.Dictionary(corpus)
bow_corpus = [dic.doc2bow(doc) for doc in corpus]


# In[ ]:


#To build the LDA topic model using LdaModel(), you need the corpus and the dictionary. Let’s create them first and then build the model.


# In[49]:


lda_model = gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 4, 
                                   id2word = dic,                                    
                                   passes = 10,
                                   workers = 2)
lda_model.show_topics()


# In[62]:


pip install pyLDAvis 


# In[68]:


import pyLDAvis.gensim_models as gensimvis


# In[69]:


def plot_lda_vis(lda_model, bow_corpus, dic):
    pyLDAvis.enable_notebook()
    vis = gensimvis.prepare(lda_model, bow_corpus, dic)
    return vis


# In[70]:


plot_lda_vis(lda_model, bow_corpus, dic)


# In[ ]:


#pyLDAVis is the most commonly used and a nice way to visualise the information contained in a topic model.
#Each bubble represents a topic. The larger the bubble, the higher percentage of the number of tweets in the corpus is about that topic.
#Blue bars represent the overall frequency of each word in the corpus. If no topic is selected, the blue bars of the most frequently used words will be displayed.
#Red bars give the estimated number of times a given term was generated by a given topic.The further the bubbles are away from each other, the more different they are.
#On the left side, the area of each circle represents the importance of the topic relative to the corpus. As there are four topics, we have four circles.
#The distance between the center of the circles indicates the similarity between the topics. If any two topics overlap, this indicates that the topics are more similar.
#On the right side, the histogram of each topic shows the top 30 relevant words.  
#A good topic model will have big and non-overlapping bubbles scattered throughout the chart,and from our chart we can see no two bubbles are overlapping. 


# In[ ]:





# In[ ]:


#Extracting the most common words


# In[189]:


all_words = []
for line in list(df['text']):
    words = line.split()
    for word in words:
        all_words.append(word.lower())
    
    
Counter(all_words).most_common(10)


# In[ ]:


#In the cell above we extracted the most common words in the dataset and listed the top ten.
#We encounter words like i, and & is, as they are very highly used in human expressions. These kind of words usually appear equally in both negative and positive oriented expressions and as such they bring very little information that can be incorporated in the model.


# In[191]:


# plot word frequency distribution of first few words
plt.figure(figsize=(12,5))
plt.title('Top 30 most common words')
plt.xticks(fontsize=13, rotation=90)
fd = nltk.FreqDist(all_words)
fd.plot(30,cumulative=False)
# log-log plot
word_counts = sorted(Counter(all_words).values(), reverse=True)
plt.figure(figsize=(12,5))
plt.loglog(word_counts, linestyle='-', linewidth=1.5)
plt.ylabel("Freq")
plt.xlabel("Word Rank")
plt.title('log-log plot of words frequency')


# In[ ]:


#First one is the graph showing the frequency of the first 30 words.
#Here is a log-log plot for the words frequency which is similar to the previous frequency graph but includes all words and is plotted on a base 10 logarithmic scale which helps us visualize the rapidly diminishing frequency of words as their rank drops.


# In[ ]:




