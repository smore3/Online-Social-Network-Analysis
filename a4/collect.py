from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
import configparser
import pickle
import re

consumer_key = 'E2BBWsM2okOz6fgWUwQK7oTHh'
consumer_secret = 'PVa441I8ZLVFIw0Jru3lHscXv0rxjsverRz0VhMB98hbL5LKm6'
access_token = '769784365609947136-6AqpFdwjd5w0RIFx4OxM4as462u4kMw'
access_token_secret = '6SioQzV3ngaG8uMG3me6n3Bs59oJ6J4zrh9p0EuQJdCS6'

# This method is done for you.
def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)

def calculate_min_id(tweets):
    #This method is to make sure that the tweets are most recent ones
    min_id = tweets[0]['id']
    for i in range(len(tweets)):
        if min_id > tweets[i]['id']:
            min_id = tweets[i]['id']
    return min_id

def get_tweets(twitter):
    tweets=[]
    num_queries=10
    query={'q': 'DearZindagi', 'count': 100, 'lang': 'en'}
    for tweet in robust_request(twitter, 'search/tweets', query):
            tweets.append(tweet)

    min_id=calculate_min_id(tweets)
    for n in range(num_queries):
        t=[]
        query = {'q': 'DearZindagi', 'count': 100, 'lang': 'en', 'max_id': min_id-1}

        for tweet in robust_request(twitter, 'search/tweets', query):
            check1 = all(x in tweet['user']['screen_name'].lower() for x in ['Gauri', 'Shinde']) #removing the accounts belonging to the celebrities
            check2 = all(x in tweet['user']['screen_name'].lower() for x in ['Alia', 'Bhatt'])   #removing the accounts belonging to the celebrities
            if 'srk' not in tweet['user']['screen_name'].lower() and not check1 and not check2:
                t.append(tweet)
                tweets.extend(t)
                min_id=calculate_min_id(t)

                pickle.dump(tweets, open('tweets.pkl', 'wb'))
    #remove urls from tweets
    for t in tweets:
        t['text'] = re.sub('http\S+', '', t['text'])

    print("sample tweet:",tweets[50]['text'])
    #print([tweets[i]['text'] for i in range(10)])

    #dump data into pickle

    return tweets

def main():

    twitter = get_twitter()
    print('Established Twitter connection.')
    print('Getting tweets')
    tweets=get_tweets(twitter)
    print('%d tweets collected' %len(tweets))

if __name__ == '__main__':
    main()