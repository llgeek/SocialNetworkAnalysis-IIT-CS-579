"""
classify.py
"""
import re
import json
import networkx as nx
from collections import Counter, defaultdict, deque
import sys
import glob
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen



def read_data(path):
    return json.load(open(path + 'tweets.txt'))

def show_tweets_stat(tweets):
    for key, value in tweets.items():
        print("Account %s: %d tweets" %(key, len(value)))    


def download_afinn():
    url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
    zipfile = ZipFile(BytesIO(url.read()))
    afinn_file = zipfile.open('AFINN/AFINN-111.txt')
    afinn = dict()

    for line in afinn_file:
        parts = line.strip().split()
        if len(parts) == 2:
            afinn[parts[0].decode("utf-8")] = int(parts[1])
    print("read %d AFINN terms." %(len(afinn)))
    return afinn



# def tokenize(text, keep_internal_punct=False):
#     if keep_internal_punct == False:
#         return np.array(re.findall('[\w_]+', text.lower()))
#     else:
#         return np.array(re.findall('[\w_][^\s]*[\w_]|[\w_]', text.lower()))

def tokenize(text):
    return re.sub('\W+', ' ', text.lower()).split()


def afinn_posneg(terms, afinn, verbose=False):
    pos = 0
    neg = 0
    for t in terms:
        if t in afinn:
            if verbose:
                print('\t%s=%d' % (t, afinn[t]))
            if afinn[t] > 0:
                pos += afinn[t]
            else:
                neg += -1 * afinn[t]
    return pos, neg


def afinn_sentiment(tweets, afinn):
    for u in tweets:
        for term in tweets[u]:
            term['pos'], term['neg'] = afinn_posneg(term['tokens'], afinn)


def print_top10_pos(tweets):
    for u in tweets:
        print("\nfor user %s:" %u)
        print("pos\t\tneg\t\ttweets")
        templist = sorted(tweets[u], key = lambda x : (-x['pos'], x['neg']))
        for term in templist[:10]:
            print(term['pos'], term['neg'], term['text'])


def print_top10_neg(tweets):    
    for u in tweets:
        print("for user %s:" %u)
        print("pos\t\tneg\t\ttweets")
        templist = sorted(tweets[u], key = lambda x : (-x['neg'], x['pos']))
        for term in templist[:10]:
            print(term['pos'], term['neg'], term['text'])


def main():
    old_data_path = "./old_tweets/"
    new_data_path = "./new_tweets/"

    print("download AFINN file...")
    afinn = download_afinn()

    choice = input("\nRead original tweets or newly downloaded tweets? \n 1: original \t\t 2: new")
    while (choice != '1' and choice != '2'):
        choice = input("Only accept '1' or '2', try again. \n 1: original \t\t 2: new")

    readtweets = dict()
    print("Loading data...")
    if choice == '1':
        readtweets = read_data(old_data_path)
    else:
        readtweets = read_data(new_data_path)

    print("Show statistics of tweets crawled")
    show_tweets_stat(readtweets)

    tweets = defaultdict()
    print("Tokenize tweets...")
    for u in readtweets:
        textlist = []
        for text in readtweets[u]:
            textlist.append({'text': text, 'tokens': tokenize(text)})
        tweets[u] = textlist

    print("Simply sentiment analysis based on AFINN...")
    afinn_sentiment(tweets, afinn)

    print("print top 10 positive tweets for each user")
    print_top10_pos(tweets)

    print("\n\nprint top negative tweets for each user")
    print_top10_neg(tweets)

    json.dump(tweets, open("./sentiment_tweets.txt", 'w'))


if __name__ == "__main__":
    main()