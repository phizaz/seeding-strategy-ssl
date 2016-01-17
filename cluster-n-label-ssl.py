import alg
import evaluator
import random

import util
import sklearn.cluster
import sklearn.neighbors
from collections import Counter
import numpy

file = './datasets/iris/iris.data'
# given
cluster_count = 3
seed_count = 10
n_seeding = 5

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

expected_acc = evaluator.cluster_n_label_knn(cluster_count,
                                             seed_count,
                                             data,
                                             target,
                                             n_seeding=n_seeding)

print('expected_acc:', expected_acc)