import re
from collections import defaultdict
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import json
import pickle

def tokenize(tweet):

    if not tweet:
        return []
    tweet = tweet.lower()
    tweet = re.sub('http\S+', ' ', tweet)
    tweet = re.sub('@\S+', '', tweet)
    tweet = re.sub('#\S+', '', tweet)
    tokens = re.sub('\W+', ' ', tweet).split()
    return tokens


def get_AFINN():

    url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
    zipfile = ZipFile(BytesIO(url.read()))
    afinn_file = zipfile.open('AFINN/AFINN-111.txt')

    afinn = dict()

    for line in afinn_file:
        parts = line.strip().split()
        if len(parts) == 2:
            afinn[parts[0].decode("utf-8")] = int(parts[1])

    return afinn


def afinn_sentiment_analysis(terms, afinn):
    pos = 0
    neg = 0
    for t in terms:
        if t in afinn:
            if afinn[t] > 0:
                pos += afinn[t]
            else:
                neg += -1 * afinn[t]
    return pos, neg

def classification(tokens, tweets, afinn):
    positives = []
    negatives = []
    neutral = []
    for token_list, tweet in zip(tokens, tweets):
        pos, neg = afinn_sentiment_analysis(token_list, afinn)
        if pos > neg:
            positives.append((tweet['text'], pos, neg))
        elif neg > pos:
            negatives.append((tweet['text'], pos, neg))
        else:
            neutral.append((tweet['text'], pos, neg))

    positives = sorted(positives, key=lambda x: x[1], reverse=True)
    negatives = sorted(negatives, key=lambda x: x[2], reverse=True)
    neutral = sorted(neutral, key=lambda x: x[2])
    return positives, negatives, neutral


def main():
    tweets = pickle.load(open("tweets.pkl", "rb"))
    #download AFINN
    url = urlopen('http://www2.compute.dtu.dk/~faan/data/AFINN.zip')
    zipfile = ZipFile(BytesIO(url.read()))
    afinn_file = zipfile.open('AFINN/AFINN-111.txt')
    afinn = dict()
    for line in afinn_file:
        parts = line.strip().split()
        if len(parts) == 2:
            afinn[parts[0].decode("utf-8")] = int(parts[1])

    tokens = [tokenize(tweet['text']) for tweet in tweets]
    positives, negatives, neutral = classification(tokens, tweets, afinn)

    pickle.dump(positives, open('positive_tweets.pkl', 'wb'))
    pickle.dump(negatives, open('negative_tweets.pkl', 'wb'))
    pickle.dump(neutral, open('neutral_tweets.pkl', 'wb'))


if __name__ == '__main__':
    main()