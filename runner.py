import numpy as np
import string
import json
import pandas
import lin_reg_algs
from collections import Counter


def strip_punctuation(s):
    s.translate(None, string.punctuation)

def lower_split(s):
    return s.lower().split()

def flatten_list_of_lists(l):
    return [item for sublist in l for item in sublist]


most_common_words = []
def compute_comment_features(comment):

    # word count feature
    counter = Counter(comment['text'])
    x_count = []
    for word, count in most_common_words[:160]:
        x_count.append(counter[word])
    comment['x_count'] = x_count

    # is_root encoding
    comment['is_root'] = 0 if comment['is_root'] is False else 1

    return comment


def create_X_y(data):
    features = [compute_comment_features(comment) for comment in data]
    X_base = pandas.DataFrame(features, columns=['children', 'controversiality', 'is_root', 'x_count'])

    X_x_count = pandas.DataFrame(X_base.x_count.tolist())
    X_base = X_base.drop('x_count', axis=1)

    X_bias =pandas.DataFrame(np.ones(shape=len(data)))

    X = pandas.concat([X_base, X_x_count, X_bias], axis=1)
    y = np.array([comment['popularity_score'] for comment in data])
    return X, y


def count_most_common_words(data):
    comments_combined = flatten_list_of_lists([comment['text'] for comment in data])
    counter = Counter(comments_combined)
    return counter.most_common()


def preprocess_routine(comment):
    comment['text'] = lower_split(comment['text'])
    return comment


if __name__ == "__main__":

    with open("proj1_data.json") as fp:
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

    X_train, y_train = create_X_y(train)
    X_validate, y_validate = create_X_y(validate)
    X_test, y_test = create_X_y(test)

    # lin_reg_algs.closed_form(X_train, y_train)
    # lin_reg_algs.gradient_descent(X_train, y_train, beta='', eta='', epsilon='')






