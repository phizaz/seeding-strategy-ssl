import alg
import random

import util
import sklearn.cluster
import sklearn.neighbors
from collections import Counter
import numpy

file = './datasets/iris/iris.data'
# given
cluster_count = 3
n_seeding = 5
seed_count = 40

dataset = util.load_data(file)

def remove_label(data):
    return map(lambda x: x[:-1],
               data)

data = remove_label(dataset)
data = util.to_number(data)
data = util.to_list(data)
data = util.rescale(data)

def get_label(data):
    return map(lambda x: x[-1],
               data)

target = get_label(dataset)
target = util.to_list(target)

goodK = util.goodKforKNN(data, target)
print('good K:', goodK)

accuracies = []
for nth_seeding in range(n_seeding):
    knn_ssl = alg.cluster_n_label_classifier(cluster_count=cluster_count,
                                             seed_count=seed_count,
                                             goodK=goodK)
    scores = sklearn.cross_validation.cross_val_score(knn_ssl, data, target,cv=5)

    print('scores:', scores)
    acc = scores.mean()
    accuracies.append(acc)

expected_acc = numpy.array(accuracies).mean()
print('expected acc:', expected_acc)
