import csv

import sklearn.neighbors

import numpy
import random
from collections import Counter
from itertools import chain
from operator import itemgetter


def load_data(file):
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

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

def evaluator(classifer, test_data, test_label):
    total = len(test_data)
    score = 0

    for i in range(total):
        prediction = classifer.predict(test_data)
        if (prediction == test_label[i]):
            score += 1

    return score, total

def crossvalidate(n_fold, data, label, fn):

    shuffled_list = random_list(len(data))
    testing_size = len(data) / n_fold

    accuracies = []
    for fold in range(n_fold):

        testing_start = fold * testing_size

        testing_list = []
        training_list = []
        for i in range(0, len(data)):
            if i >= testing_start and i < testing_start + testing_size:
                testing_list.append(shuffled_list[i])
            else:
                training_list.append(shuffled_list[i])

        train_data = [data[i] for i in training_list]
        train_label = [label[i] for i in training_list]

        test_data = [data[i] for i in testing_list]
        test_label = [data[i] for i in testing_list]

        classifier = fn(train_data, train_label)

        score, total = evaluator(classifier, test_data, test_label)
        accuracies.append(score / total)

    return numpy.array(accuracies)

def goodKforKNN(data, label, weights = 'uniform'):
    search_range = filter(lambda x: x % 2,
                          range(1, int(len(data) ** 0.5) + 1))

    accuracies = []
    for k in search_range:
        print('testing k:', k)
        knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k,
                                                     weights=weights)
        scores = sklearn.cross_validation.cross_val_score(knn,
                                                          data,
                                                          label,
                                                          cv = 5)
        accuracy = scores.mean()
        print('acc:', accuracy)
        accuracies.append((k, accuracy))

    best_k, _ = max(accuracies, key=itemgetter(1))

    return best_k