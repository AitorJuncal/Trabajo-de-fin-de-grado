import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import contractions
import string
import os
import random
import codecs

def read_sentiment(dataset):
    file = pd.read_csv("Sentiment/"+dataset+"/train.csv", encoding="latin", names=["Text", "Class"])
    file = file.sample(frac=1)
    tweets_train=file['Text'].tolist()
    clases_train=file['Class'].tolist()
    for i in range(len(clases_train)):
        if clases_train[i] == 4:
            clases_train[i] = 1

    file = pd.read_csv("Sentiment/"+dataset+"/test.csv", encoding="latin", names=["Text", "Class"])
    file = file.sample(frac=1)
    tweets_test=file['Text'].tolist()
    clases_test=file['Class'].tolist()
    for i in range(len(clases_test)):
        if clases_test[i] == 4:
            clases_test[i] = 1

    return tweets_train, tweets_test, clases_train, clases_test

def read_amazon(dataset):
    file = pd.read_csv("Amazon/"+dataset+"/train.csv", encoding="latin", names=["Text", "Class"])
    file = file.sample(frac=1)
    tweets_train=file['Text'].tolist()
    clases_train=file['Class'].tolist()

    file = pd.read_csv("Amazon/"+dataset+"/test.csv", encoding="latin", names=["Text", "Class"])
    file = file.sample(frac=1)
    tweets_test=file['Text'].tolist()
    clases_test=file['Class'].tolist()

    return tweets_train, tweets_test, clases_train, clases_test

def read_t4sa(dataset):
    file = pd.read_csv("T4SA/"+dataset+"/train.csv", encoding="latin", names=["Text", "Class"])
    file = file.sample(frac=1)
    tweets_train=file['Text'].tolist()
    clases_train=file['Class'].tolist()

    file = pd.read_csv("T4SA/"+dataset+"/test.csv", encoding="latin", names=["Text", "Class"])
    file = file.sample(frac=1)
    tweets_test=file['Text'].tolist()
    clases_test=file['Class'].tolist()

    return tweets_train, tweets_test, clases_train, clases_test

def read_dataset(dataset):
    path_train_neg = dataset+"/train/neg"
    path_train_pos = dataset+"/train/pos"
    path_test_neg = dataset+"/test/neg"
    path_test_pos = dataset+"/test/pos"
    files_train_neg = os.listdir(path_train_neg)
    files_train_pos = os.listdir(path_train_pos)
    files_test_neg = os.listdir(path_test_neg)
    files_test_pos = os.listdir(path_test_pos)

    tweets_train_neg = read_tweets(path_train_neg, files_train_neg)
    tweets_train_pos = read_tweets(path_train_pos, files_train_pos)
    tweets_test_neg = read_tweets(path_test_neg, files_test_neg)
    tweets_test_pos = read_tweets(path_test_pos, files_test_pos)

    tweets_train = tweets_train_neg + tweets_train_pos
    random.shuffle(tweets_train)
    tweets_test = tweets_test_neg + tweets_test_pos
    random.shuffle(tweets_test)

    clases_train = []
    for i in range(len(tweets_train)):
        tweet = tweets_train[i]
        if tweet in tweets_train_neg:
            clases_train.append(0)
        else:
            clases_train.append(1)

    clases_test = []
    for i in range(len(tweets_test)):
        tweet = tweets_test[i]
        if tweet in tweets_test_neg:
            clases_test.append(0)
        else:
            clases_test.append(1)

    return tweets_train, tweets_test, clases_train, clases_test


def read_tweets(path, files):
    tweets = []
    for file in files:
        f = codecs.open(path+"/"+file, 'r', 'utf-8')
        tweet = f.read()
        tweets.append(tweet)
    return tweets


def preprocess(tweets, N):
    """Borramos los caracteres innecesarios"""
    for i in range(0, N):
        #Pasamos el tweet a string.
        tweets[i] = str(tweets[i])
        #ponemos la frase en minúsculas.
        tweets[i] = tweets[i].lower()
        #borramos los caracteres hexadecimales
        tweets[i] = re.sub(r'[^\x00-\x7f]',r'', tweets[i]) 
        #Borramos hashtags
        tweets[i] = re.sub(r'(@\S+)', r'', tweets[i])
        #Borramos las menciones @
        tweets[i] = re.sub(r'(#\S+)', r'', tweets[i])
        #Borramos los links
        tweets[i] = re.sub(r'(http\S+) | (https\S+)', r'', tweets[i])
        
        #Expandimos las contracciones del ingles.
        expanded_sentence = []
        for word in tweets[i].split():
            expanded_sentence.append(contractions.fix(word))
        tweets[i] = ' '.join(expanded_sentence)    
            
        #Borramos las stopwords
        stop_words = set(stopwords.words('english'))
        filtered_sentence = []
        for word in tweets[i].split():
            if word not in stop_words:
                filtered_sentence.append(word)       
        tweets[i] = ' '.join(filtered_sentence)
        
        #Borramos los puntos suspensivos.
        tweets[i] = re.sub('[...]', ' ', tweets[i])

    #ahora podemos borrar los signos de puntuacion, dividir el tweet en palabras y
    #obtener su tag.
    tagged_tweets = []
    for i in range(0, N):
        tweets[i] = re.sub('['+string.punctuation+']', ' ', tweets[i])
        """Entities extraction"""
        tagged_tweets.append(nltk.pos_tag(nltk.word_tokenize(tweets[i])))
    tweets = tagged_tweets
        

    """Borramos las preposiciones"""
    preposicion_tag = 'IN'
    for tweet in tweets:
        for word in tweet:
            if word[1] == preposicion_tag:
                tweet.remove(word)
    
    return tweets
