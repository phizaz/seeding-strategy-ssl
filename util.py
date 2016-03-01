import csv

import sklearn.neighbors

import numpy
import numpy as np
import random
from collections import Counter
from itertools import chain
from operator import itemgetter
from sklearn.neighbors import KNeighborsClassifier


def load_data(file, delimiter=','):
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=delimiter, quotechar='|')

        result = []
        for row in spamreader:
            if len(row) > 0:
                result.append(row)
        return result

def to_number(data):
    return map(lambda row: map(lambda col: float(col),
                               row),
               data)

def rescale(data):
    dataT = numpy.array(data).transpose()

    def rescale_row(row):
        maximum = max(row)
        minimum = min(row)

        return map(lambda col: (col - minimum) / (maximum - minimum),
                   row)

    dataT = map(rescale_row, dataT)

    dataT = to_list(dataT)
    data = numpy.array(dataT).transpose()
    return data


def to_list(data):
    if hasattr(data, '__iter__') and not isinstance(data, str):
        return list(map(lambda x: to_list(x),
                        data))
    else:
        return data

def is_prime(n):
    if n == 2 or n == 3: return True
    if n < 2 or n%2 == 0: return False
    if n < 9: return True
    if n%3 == 0: return False
    r = int(n**0.5)
    f = 5
    while f <= r:
        if n%f == 0: return False
        if n%(f+2) == 0: return False
        f +=6
    return True

def random_list(n):
    l = [i for i in range(n)]
    for i in range(n):
        rand = random.randint(0, n-1)
        if rand != i:
            l[i], l[rand] = l[rand], l[i]
    return l

    # generate array with a given size (r) containig random members in range (0, n - 1)
    def seed_numbers(n, r):
        return random_list(n)[:r]

# given an array find the most common
# if it has mnay most_commons return randomly amongst them
# return that pair (name, freq)
def most_common(cluster):
    count = Counter(cluster).most_common()
    max_count = count[0][1]
    most_commons = list(filter(lambda x: x[1] == max_count,
                               count))
    result = most_commons[random.randint(0, len(most_commons) - 1)]
    return result

def good_K_for_KNN_with_testdata(X, Y, X_test, Y_test):
    accuracies = []
    descending_cnt = 0
    descending_threshold = 15

    for k in range(1, len(X) + 1):
        # print('k:', k)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X, Y)
        Y_pred = knn.predict(X_test)
        score = sum(np.array_equal(a, b) for a, b in zip(Y_pred, Y_test))
        # print('acc:', score)
        accuracies.append((k, score))
        if k > 1:
            current_acc = score
            _, before_acc = accuracies[-2]
            descending_cnt += current_acc <= before_acc

            if descending_cnt >= descending_threshold:
                break

    best_k, best_acc = max(accuracies, key=itemgetter(1))
    return best_k, best_acc


def goodKforKNN(data, label, weights = 'uniform'):
    search_range = filter(lambda x: x % 2,
                          range(1, int(len(data) ** 0.5) + 1))

    accuracies = []
    worse_count = 0
    worse_threshold = 15
    for k in search_range:
        print('testing k:', k)
        knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k,
                                                     weights=weights)
        scores = sklearn.cross_validation.cross_val_score(knn,
                                                          data,
                                                          label,
                                                          cv = 5,
                                                          n_jobs = 2)
        accuracy = scores.mean()
        print('acc:', accuracy)

        if len(accuracies) > 0 and accuracy < accuracies[-1][1]:
            worse_count += 1
        else:
            worse_count = 0

        accuracies.append((k, accuracy))

        if worse_count >= worse_threshold:
            break

    best_k, accuracy = max(accuracies, key=itemgetter(1))

    return best_k, accuracy

def requires(list, dict):
    result = []
    for field in list:
        if field not in dict:
            raise Exception('no ' + field)
        result.append(dict[field])

    return result