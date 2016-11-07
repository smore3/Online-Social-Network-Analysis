# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    movies['tokens']=movies['genres'].apply(lambda x:tokenize_string(x))
    return movies


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    ###TODO
    movies['features']=""
    """""Finding unique features from all the tokens"""
    features_list=sorted(set(term for terms in movies.tokens for term in terms))

    """"Calcualting vocab using the features_list"""
    index = 0
    vocab = defaultdict(lambda: 0)
    for term in features_list:
        vocab[term]=index
        index+=1

    """"Calcualting IDF for every term in the features_list"""
    IDF = defaultdict(lambda: 0)
    for feature in features_list:
        count=0
        for doc in movies.tokens:
            if feature in doc:
                count+=1
        IDF[feature]=math.log(movies.shape[0]/count)

    """"Calcualting the csr matrix for every row"""
    for index,row in movies.iterrows():
        csr_row,csr_col,csr_data,term_freq_doc=[],[],[],Counter()
        max_freq=0
        tfidf=defaultdict(lambda :0)
        term_freq_doc=Counter(row.tokens)
        max_freq=max(term_freq_doc.values())

        for term in row.tokens:
            tfidf[term]=(term_freq_doc[term]/max_freq)*(IDF[term])
            if term in features_list:
                [csr_row.append(0)]
                [csr_col.append(vocab[term])]
                [csr_data.append(tfidf[term])]

        movies.set_value(index,'features',csr_matrix((csr_data, (csr_row,csr_col)), shape=(1,len(vocab)))) #setting the calculated csr matrix in new colum of every row

    return movies, vocab


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO
    a=a.todense()
    b=b.todense()
    a=np.array(a)
    b=np.array(b)
    
    cosine = float(np.dot(a, b.T) / np.linalg.norm(a) / np.linalg.norm(b))
    #print(cosine)
    return round(cosine,5)


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO
    predicted_rating=[]
    movies = movies.set_index(['movieId'])
    for index,row in ratings_test.iterrows():
        movie_to_predict=row.movieId
        movies_per_userId = ratings_train[ratings_train.userId == row.userId] #all movies per user in ratings_test
        #print(movies_per_userId)
        cos_sim,sum,ratings=0.0,0.0,0.0
        for index2,row2 in movies_per_userId.iterrows():
            user_movie_id=row2.movieId					     
            feature_col_index=3
            compare_movie_csr=movies.loc[movie_to_predict][feature_col_index]  #csr matrix for the movie to be predicted

            temp=cosine_sim(compare_movie_csr, movies.loc[user_movie_id][feature_col_index]) #cosine similarity between movie to be predicted and other movies that particular user rated
            if temp>=0:
                cos_sim+=(temp*row2.rating)
                sum+=temp
            ratings+=row2.rating

        if sum==0:						#if cosine similarity if zero for all movies then average the ratings
            weighted_avg=ratings/movies_per_userId.shape[0]
        else:
            weighted_avg=cos_sim/sum
        #print(weighted_avg)
        predicted_rating.append(round(weighted_avg,1))

    return np.array(predicted_rating)


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
