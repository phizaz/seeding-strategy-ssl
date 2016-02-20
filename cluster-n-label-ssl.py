import alg
import evaluator
import random

import util
import sklearn.cluster
import sklearn.neighbors
from collections import Counter
import numpy

file = './datasets/iris/iris.data'
#file =  './datasets/pendigits/pendigits.tra'
#file = './datasets/satimage/sat.trn'
#file = './datasets/letter-recognition/letter-recognition.data'
# given

dataset = util.load_data(file, delimiter=',')

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

#cluster_count_list = [26, 52, 104, 208, 416, 832]
cluster_count_list = [3]

for cluster_count in cluster_count_list:
    print('cluster count:', cluster_count)
# cluster_count = 8
n_seeding = 5
seed_count = int(len(data) * 0.1)

expected_acc = evaluator.cluster_n_label_knn(cluster_count,
                                             seed_count,
                                             data,
                                             target,
                                             n_seeding=n_seeding)

print('expected_acc:', expected_acc)