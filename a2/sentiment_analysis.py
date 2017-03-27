# coding: utf-8

"""
CS579: Assignment 2

In this assignment, you will build a text classifier to determine whether a
movie review is expressing positive or negative sentiment. The data come from
the website IMDB.com.

You'll write code to preprocess the data in different ways (creating different
features), then compare the cross-validation accuracy of each approach. Then,
you'll compute accuracy on a test set and do some analysis of the errors.

The main method takes about 40 seconds for me to run on my laptop. Places to
check for inefficiency include the vectorize function and the
eval_all_combinations function.

Complete the 14 methods below, indicated by TODO.

As usual, completing one method at a time, and debugging with doctests, should
help.
"""

# No imports allowed besides these.
from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request
from time import time


def download_data():
    """ Download and unzip data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/xk4glpk61q3qrg2/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


def read_data(path):
    """
    Walks all subdirectories of this path and reads all
    the text files and labels.
    DONE ALREADY.

    Params:
      path....path to files
    Returns:
      docs.....list of strings, one per document
      labels...list of ints, 1=positive, 0=negative label.
               Inferred from file path (i.e., if it contains
               'pos', it is 1, else 0)
    """
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. E.g., in the example below, the token "isn't"
    is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.

    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.

    >>> tokenize(" Hi there! Isn't this fun?", keep_internal_punct=False)
    array(['hi', 'there', 'isn', 't', 'this', 'fun'],
          dtype='<U5')
    >>> tokenize("Hi there! Isn't this fun? ", keep_internal_punct=True)
    array(['hi', 'there', "isn't", 'this', 'fun'],
          dtype='<U5')
    """
    ###TODO
    if keep_internal_punct:
        #arr = re.compile("[^\w'|-]").sub(" ",doc).lower()
        arr = doc.lower().split()
        for i in range(len(arr)):
            arr[i] = re.sub('^\W+', '', re.sub('\W+$', '', arr[i].strip()).strip())
        return np.array(arr)
    else:
        arr = re.sub('[^A-Za-z0-9_]+', ' ', doc.lower().strip())
        return np.array(arr.split())

def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_features(['hi', 'there', 'hi'], feats)
    >>> sorted(feats.items())
    [('token=hi', 2), ('token=there', 1)]
    """
    ###TODO

    for word in tokens:
        key="".join(("token=",word))
        feats[key] +=1


def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.

    For example [a, b, c, d] with k=3 will consider the
    windows: [a,b,c], [b,c,d]. In the first window,
    a_b, a_c, and b_c appear; in the second window,
    b_c, c_d, and b_d appear. This example is in the
    doctest below.
    Note that the order of the tokens in the feature name
    matches the order in which they appear in the document.
    (e.g., a__b, not b__a)

    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
    >>> sorted(feats.items())
    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 1)]
    """
    ###TODO
    windows,pairs, temp = [], [], []
    for j in range(len(tokens) - (k - 1)):
        temp = tokens[j:j + k]
        windows.append(temp)

    for i in windows:
        for j in combinations(i, 2):
            key = list(j)
            pairs.append(key)

    for i in pairs:
        key = i[0] + '__' + i[1]
        feats["".join(("token_pair=",key))] +=1


neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])

def lexicon_features(tokens, feats):
    """
    Add features indicating how many time a token appears that matches either
    the neg_words or pos_words (defined above). The matching should ignore
    case.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    In this example, 'LOVE' and 'great' match the pos_words,
    and 'boring' matches the neg_words list.
    >>> feats = defaultdict(lambda: 0)
    >>> lexicon_features(np.array(['i', 'LOVE', 'this', 'great', 'boring', 'movie']), feats)
    >>> sorted(feats.items())
    [('neg_words', 1), ('pos_words', 2)]
    """
    ###TODO
    #print(neg_words, pos_words)
    feats['neg_words'] = 0
    feats['pos_words'] = 0
    for token in tokens:
        if token.lower() in neg_words:
            feats['neg_words'] +=1
        elif token.lower() in pos_words:
            feats['pos_words'] +=1


def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.

    Params:
      tokens........array of token strings from a document.
      feature_fns...a list of functions, one per feature
    Returns:
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.

    >>> feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])
    >>> feats
    [('neg_words', 0), ('pos_words', 2), ('token=LOVE', 1), ('token=great', 1), ('token=i', 1), ('token=movie', 1), ('token=this', 1)]
    """
    ###TODO
    feats=defaultdict(lambda :0)
    for feature in feature_fns:
        feature(tokens, feats)

    return sorted(feats.items())


def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    Given the tokens for a set of documents, create a sparse
    feature matrix, where each row represents a document, and
    each column represents a feature.

    Params:
      tokens_list...a list of lists; each sublist is an
                    array of token strings from a document.
      feature_fns...a list of functions, one per feature
      min_freq......Remove features that do not appear in
                    at least min_freq different documents.
    Returns:
      - a csr_matrix: See https://goo.gl/f5TiF1 for documentation.
      This is a sparse matrix (zero values are not stored).
      - vocab: a dict from feature name to column index. NOTE
      that the columns are sorted alphabetically (so, the feature
      "token=great" is column 0 and "token=horrible" is column 1
      because "great" < "horrible" alphabetically),

    >>> docs = ["Isn't this movie great?", "Horrible, horrible movie"]
    >>> tokens_list = [tokenize(d) for d in docs]
    >>> feature_fns = [token_features]
    >>> X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
    >>> type(X)
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> X.toarray()
    array([[1, 0, 1, 1, 1, 1],
           [0, 2, 0, 1, 0, 0]], dtype=int64)
    >>> sorted(vocab.items(), key=lambda x: x[1])
    [('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]
    """
    ###TODO
    features=[]
    #start=time()
    doc_counter,features_dict=0,{}
    for doc in tokens_list:
        feature=featurize(doc,feature_fns)
        features_dict[doc_counter]=feature
        features+=feature
        doc_counter+=1
    #print(features_dict.items())

    min_freq_dict=Counter()

    #print("Total features function time:",(time()-start))

    for feature in features:
        feature_value, count = feature
        min_freq_dict[feature_value] +=1

    count=0
    #start2=time()
    if vocab==None:
        final_features = []
        for key, value in min_freq_dict.items():
            if value >= min_freq:
                final_features.append(key)

        final_features.sort()
        vocab = {}
        for key in final_features:
            vocab[key] = count
            count+=1


    #print("vocab function time:", (time() - start2))
    #start3=time()
    row,col,data,freq=[],[],[],{}
    for row_num in range(len(tokens_list)):
        for feature_value,count in features_dict[row_num]:
            if feature_value in vocab.keys():
                [row.append(row_num)]
                [col.append(vocab[feature_value])]
                [data.append(count)]

    X=csr_matrix((data, (row,col)))
    #print("Total main appending time:", (time() - start3))
    return X, vocab


def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).

    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.

    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    ###TODO
    kf = KFold(len(labels), k)
    counter = 0
    accuracies = []
    for train, test in kf:
        clf.fit(X[train], labels[train])
        predicted = clf.predict(X[test])
        acc = accuracy_score(labels[test], predicted)
        accuracies.append(acc)
        counter += 1
    return np.mean(accuracies)


def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.

    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.

    In addition to looping over possible assignments to
    keep_internal_punct and min_freqs, we will enumerate all
    possible combinations of feature functions. So, if
    feature_fns = [token_features, token_pair_features, lexicon_features],
    then we will consider all 7 combinations of features (see Log.txt
    for more examples).

    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])

    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.

      This list should be SORTED in descending order of accuracy.

      This function will take a bit longer to run (~20s for me).
    """
    ###TODO
    feature_funcslist=[]
    #dict_punct, dict_features, dict_min_freq, dict_accuracy={},{},{},{}

    for n in range(1, len(feature_fns)+1):
        for sub_feature_list in list(combinations(list(feature_fns),n)):
            feature_funcslist.append(sub_feature_list)
    #print(feature_funcslist)
    final_list=[]
    token_list1=[]
    token_list2 = []
    for doc in docs:
        token_list1.append(tokenize(doc,keep_internal_punct=True))
    for doc in docs:
        token_list2.append(tokenize(doc,keep_internal_punct=False))

    clf=LogisticRegression()
    for value in punct_vals:
        for freq in min_freqs:
            for feature in feature_funcslist:
                final_dict = {}
                #start=time()
                if value==True:
                    X, vocab = vectorize(token_list1,feature,freq)
                    #print("Vectorize time: ", (time() - start))
                else:
                    X, vocab = vectorize(token_list2, feature, freq)
                    #print("Vectorize time: ", (time() - start))
                acc=cross_validation_accuracy(clf,X,labels,5)
                final_dict['punct'], final_dict['features'], final_dict['min_freq'], final_dict['accuracy'] = value, feature, freq, acc
                final_list.append(final_dict)
                #print(final_dict.items())
    result=sorted(final_list,key=lambda x:x['accuracy'], reverse=True)
    return result

def plot_sorted_accuracies(results):
    """
    Plot all accuracies from the result of eval_all_combinations
    in ascending order of accuracy.
    Save to "accuracies.png".
    """
    ###TODO
    acc_list=[]
    for value in results:
        acc_list.append(value['accuracy'])

    plt.plot(sorted(acc_list))
    plt.xlabel('settings')
    plt.ylabel('accuracy')
    plt.savefig('accuracies.png')


def mean_accuracy_per_setting(results):
    """
    To determine how important each model setting is to overall accuracy,
    we'll compute the mean accuracy of all combinations with a particular
    setting. For example, compute the mean accuracy of all runs with
    min_freq=2.

    Params:
      results...The output of eval_all_combinations
    Returns:
      A list of (accuracy, setting) tuples, SORTED in
      descending order of accuracy.
    """
    ###TODO
    dict_setting1 = defaultdict(lambda :0)
    dict_setting2 = defaultdict(lambda :0)
    dict_setting3 = defaultdict(lambda :0)
    list_settings = []

    for value in results:

        dict_setting1.setdefault(value['punct'],[]).append(value['accuracy'])
        dict_setting2.setdefault(value['min_freq'],[]).append(value['accuracy'])
        dict_setting3.setdefault(value['features'],[]).append(value['accuracy'])

    for setting, accuracy in dict_setting1.items():
        list_settings.append((sum(accuracy)/len(accuracy), 'punct='+str(setting)))

    for setting, accuracy in dict_setting2.items():
        list_settings.append((sum(accuracy)/len(accuracy), 'min_freq='+str(setting)))

    for setting, accuracy in dict_setting3.items():
        feature_string = ''
        for i in setting:
            feature_string = feature_string + i.__name__ + ' '
        list_settings.append((sum(accuracy) / len(accuracy), 'features=' + str(feature_string.strip())))


    return sorted(list_settings, key=lambda x: x[0], reverse=True)


def fit_best_classifier(docs, labels, best_result):
    """
    Using the best setting from eval_all_combinations,
    re-vectorize all the training data and fit a
    LogisticRegression classifier to all training data.
    (i.e., no cross-validation done here)

    Params:
      docs..........List of training document strings.
      labels........The true labels for each training document (0 or 1)
      best_result...Element of eval_all_combinations
                    with highest accuracy
    Returns:
      clf.....A LogisticRegression classifier fit to all
            training data.
      vocab...The dict from feature name to column index.
    """
    ###TODO
    tokens_list=[]
    for doc in docs:
        tokens_list.append(tokenize(doc,best_result['punct']))

    X, vocab = vectorize(tokens_list,best_result['features'],best_result['min_freq'])
    clf=LogisticRegression()
    clf.fit(X,labels)

    return clf, vocab


def top_coefs(clf, label, n, vocab):
    """
    Find the n features with the highest coefficients in
    this classifier for this label.
    See the .coef_ attribute of LogisticRegression.

    Params:
      clf.....LogisticRegression classifier
      label...1 or 0; if 1, return the top coefficients
              for the positive class; else for negative.
      n.......The number of coefficients to return.
      vocab...Dict from feature name to column index.
    Returns:
      List of (feature_name, coefficient) tuples, SORTED
      in descending order of the coefficient for the
      given class label.
    """
    ###TODO

    coef=clf.coef_[0:]
    feature_coef=[]
    vocab_dict_reverse={}

    for key,value in vocab.items():
        vocab_dict_reverse[value]=key

    indices_coef = [i for i in sorted(enumerate(coef[0]), key=lambda x:x[1])]

    if label==1:
        for index, coefficient in indices_coef[::-1][:n]:
            feature_coef.append((vocab_dict_reverse[index],float(coefficient)))

    elif label==0:
        for index,coefficient in indices_coef[:n]:
            feature_coef.append((vocab_dict_reverse[index], float(-1*coefficient)))

    return feature_coef


def parse_test_data(best_result, vocab):
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab should
    be passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.

    Note: use read_data function defined above to read the
    test data.

    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
    """
    ###TODO
    test_docs, test_labels = read_data(os.path.join('data', 'test'))
    tokens_list = []

    for doc in test_docs:
        tokens_list.append(tokenize(doc, best_result['punct']))

    X_test, vocab = vectorize(tokens_list, best_result['features'], best_result['min_freq'], vocab)

    return test_docs, test_labels, X_test


def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    """
    Print the n testing documents that are misclassified by the
    largest margin. By using the .predict_proba function of
    LogisticRegression <https://goo.gl/4WXbYA>, we can get the
    predicted probabilities of each class for each instance.
    We will first identify all incorrectly classified documents,
    then sort them in descending order of the predicted probability
    for the incorrect class.
    E.g., if document i is misclassified as positive, we will
    consider the probability of the positive class when sorting.

    Params:
      test_docs.....List of strings, one per test document
      test_labels...Array of true testing labels
      X_test........csr_matrix for test data
      clf...........LogisticRegression classifier fit on all training
                    data.
      n.............The number of documents to print.

    Returns:
      Nothing; see Log.txt for example printed output.
    """
    ###TODO
    predicted=clf.predict(X_test)
    pred_prob=clf.predict_proba(X_test)
    incorrect_doc_list=[]

    for i in range(len(test_labels)):
        if test_labels[i] != predicted[i]:
            if predicted[i]==1:
                incorrect_doc_list.append((test_labels[i], predicted[i], pred_prob[i][1], test_docs[i]))
            else:
                incorrect_doc_list.append((test_labels[i], predicted[i], pred_prob[i][0], test_docs[i]))

    incorrect_doc_list.sort(key= lambda x:x[2], reverse=True)

    for i in range(n):
        print('\ntruth=' + str(incorrect_doc_list[i][0]) + ' predicted=' + str(incorrect_doc_list[i][1]) + ' proba=' + str(incorrect_doc_list[i][2]))
        print(str(incorrect_doc_list[i][3]))


def main():
    """
    Put it all together.
    ALREADY DONE.
    """
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))

    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)


if __name__ == '__main__':
    main()
