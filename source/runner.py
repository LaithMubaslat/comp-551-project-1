import nltk
import numpy as np
import string
import json
import pandas
import lin_reg_algs
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import metrics

def strip_punctuation(s):
    s.translate(None, string.punctuation)

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
    for word, count in most_common_words[:160]:
        x_count.append(counter[word])
    comment['x_count'] = x_count

    # is_root encoding
    comment['is_root'] = 0 if comment['is_root'] is False else 1

    # comment length feature
    comment['length'] = len(comment)

    # most common bigrams feature:
    # currently produced a Singular X matrix, since not every common bigram will be present in the text
    # need to bias/smooth this feature
    comment_bigrams = list(nltk.collocations.BigramCollocationFinder.from_words(comment['text']).ngram_fd)
    bi_count = []
    for candidate_bigram in most_common_bigrams:
        count = comment_bigrams.count(candidate_bigram)
        bi_count.append(count)
    comment['bi_count'] = bi_count

    # other feature ideas:
    # bigram count (use NLTK)
    # comment length
    # interaction features: combinations of is_root, controversiality, children, and comment length

    return comment


def create_X_y(data):
    features = [compute_comment_features(comment) for comment in data]
    X_base = pandas.DataFrame(features,
                              columns=['children', 'controversiality', 'is_root', 'x_count', 'bi_count'])

    X_x_count = pandas.DataFrame(X_base.x_count.tolist())
    X_base = X_base.drop('x_count', axis=1)

    X_bi_count = pandas.DataFrame(X_base.bi_count.tolist())
    X_base = X_base.drop('bi_count', axis=1)

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
    finder = nltk.collocations.BigramCollocationFinder.from_words(comments_combined)
    return finder.nbest(nltk.BigramAssocMeasures.raw_freq, 160)


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
    W_train = lin_reg_algs.closed_form(X_train, y_train)
    W_val = lin_reg_algs.closed_form(X_validate, y_validate)

    y_train_results = np.array(X_train.dot(W_train))
    y_val_results = np.array(X_validate.dot(W_val))

    print('train: ' + str(metrics.mean_squared_error(y_train, y_train_results)))
    print('validation: ' + str(metrics.mean_squared_error(y_validate, y_val_results)))

    print('Gradient descent')
    alpha, epsilon, iterations = (1e-6, 1e-6, 10000)
    W_train = lin_reg_algs.gradient_descent(X_train, y_train, alpha, epsilon, iterations)
    W_val = lin_reg_algs.gradient_descent(X_validate, y_validate, alpha, epsilon, iterations)

    y_train_results = np.array(X_train.dot(W_train))
    y_val_results = np.array(X_validate.dot(W_val))

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





