import nltk
import re
import numpy as np
import string
import json
import pandas
import lin_reg_algs
import timeit
from collections import Counter
from sklearn import metrics
from sklearn.preprocessing import normalize
from nltk.corpus import stopwords

def strip_punctuation(s):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub('', s)

def remove_stopwords(word_list):
    stops = set(stopwords.words('english'))
    return [strip_punctuation(word) for word in word_list if word not in stops]

def lower_split(s):
    return s.lower().split()

def flatten_list_of_lists(l):
    return [item for sublist in l for item in sublist]

most_common_words = []
most_common_bigrams = []
def compute_comment_features(comment):

    # word count feature
    counter = Counter(comment['text'])
    x_count = []
    for word, count in most_common_words[:60]:
        x_count.append(counter[word])
    comment['x_count'] = x_count

    # is_root encoding
    comment['is_root'] = 0 if comment['is_root'] is False else 1

    # comment length feature
    comment['length'] = len(comment['text'])

    # how many of the most common bigrams present in comment? feature
    bigram_text = remove_stopwords(comment['text']) # to match the preprocessing done for bigrams
    comment_bigrams = list(nltk.collocations.BigramCollocationFinder.from_words(bigram_text).ngram_fd)
    count = 0
    for candidate_bigram in most_common_bigrams:
        count += comment_bigrams.count(candidate_bigram)
    comment['bi_count'] = count

    return comment


def create_X_y(data):
    features = [compute_comment_features(comment) for comment in data]
    X_base = pandas.DataFrame(features,
                              columns=['children', 'controversiality', 'is_root', 'x_count', 'length'])

    X_x_count = pandas.DataFrame(X_base.x_count.tolist())
    X_base = X_base.drop('x_count', axis=1)

    X_length = pandas.DataFrame(X_base.length)
    X_length_norm = (X_length - X_length.mean()) / X_length.std()
    X_base = X_base.drop('length', axis=1)

    X_bias = pandas.DataFrame({'bias': np.ones(shape=len(data))})

    X = pandas.concat([X_bias, X_base, X_x_count], axis=1)
    y = np.array([comment['popularity_score'] for comment in data])
    return X, y


def count_most_common_words(data):
    comments_combined = flatten_list_of_lists([comment['text'] for comment in data])
    counter = Counter(comments_combined)
    return counter.most_common()


def count_most_common_bigrams(data):
    comments_combined = flatten_list_of_lists([comment['text'] for comment in data])
    comments_combined = remove_stopwords(comments_combined)
    finder = nltk.collocations.BigramCollocationFinder.from_words(comments_combined)
    return finder.nbest(nltk.BigramAssocMeasures.raw_freq, 1000)


def preprocess_routine(comment):
    comment['text'] = lower_split(comment['text'])
    return comment


if __name__ == "__main__":

    with open("./data/proj1_data.json") as fp:
        data = json.load(fp)

    # STEP0: preprocess data
    data = [preprocess_routine(comment) for comment in data]

    # STEP1: separate into training, validation and test sets
    training_len = 10000
    validation_len = 1000
    train = data[0:training_len]
    validate = data[training_len:validation_len + training_len]
    test = data[validation_len + training_len:len(data)]

    most_common_words = count_most_common_words(train)
    most_common_bigrams = count_most_common_bigrams(train)

    X_train, y_train = create_X_y(train)
    X_validate, y_validate = create_X_y(validate)
    X_test, y_test = create_X_y(test)

    # STEP 2: run linear regression algorithms

    print('Closed form:')

    start_time = timeit.default_timer()
    W = lin_reg_algs.closed_form(X_train, y_train)
    print(timeit.default_timer() - start_time)

    y_train_results = np.array(X_train.dot(W))
    y_val_results = np.array(X_validate.dot(W))

    print('train: ' + str(metrics.mean_squared_error(y_train, y_train_results)))
    print('validation: ' + str(metrics.mean_squared_error(y_validate, y_val_results)))


    print('Gradient descent')
    alpha, epsilon, iterations = (1e-6, 1e-6, 10000)

    start_time = timeit.default_timer()
    W = lin_reg_algs.gradient_descent(X_train, y_train, alpha, epsilon, iterations)
    print(timeit.default_timer() - start_time)

    y_train_results = np.array(X_train.dot(W))
    y_val_results = np.array(X_validate.dot(W))

    print('train: ' + str(metrics.mean_squared_error(y_train, y_train_results)))
    print('validation: ' + str(metrics.mean_squared_error(y_validate, y_val_results)))

    # print('Ridge regression:')
    # lamb = 0.5
    # W_train = lin_reg_algs.ridge_regression(X_train, y_train, lamb)
    # W_val = lin_reg_algs.ridge_regression(X_validate, y_validate, lamb)
    #
    # y_train_results = np.array(X_train.dot(W_train))
    # y_val_results = np.array(X_validate.dot(W_val))
    #
    # print('train: ' + str(metrics.mean_squared_error(y_train, y_train_results)))
    # print('validation: ' + str(metrics.mean_squared_error(y_validate, y_val_results)))





