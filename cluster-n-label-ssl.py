import random

import util
import sklearn.cluster
from collections import Counter
import numpy

file = './datasets/iris/iris.data'
# given
cluster_count = 3
seed_count = 4

data = util.load_data(file)

def remove_label(data):
    return map(lambda x: x[:-1],
data)

train = remove_label(data)
train = util.to_number(train)
train = util.to_list(train)

def get_label(data):
    return map(lambda x: [x[-1]],
               data)

test = get_label(data)
test = util.to_list(test)

print('train:', train)
print('label:', test)

kmeans = sklearn.cluster.KMeans(n_clusters=cluster_count)
kmeans.fit(train)

seed_numbers = util.seed_numbers(len(train), seed_count)

print('seed_numbers:', seed_numbers)

clusters_labels = [ [] for i in range(cluster_count)]
seed_labels = []
for seed_number in seed_numbers:
    seed_label = test[seed_number][0]
    seed_data = train[seed_number]

    cluster = kmeans.predict([seed_data])
    clusters_labels[cluster].append(seed_label)
    seed_labels.append(seed_label)

print(seed_labels)

major_labels = map(lambda cluster: Counter(cluster).most_common()[0][0] if len(cluster) > 0 else Counter(seed_labels).most_common()[0][0],
                   clusters_labels)
major_labels = list(major_labels)

print('major:', major_labels)

score = 0
for i, data in enumerate(train):
    prediction = kmeans.predict([data])[0]
    predicted_label = major_labels[prediction]

    print('cluster:', prediction, 'label:', predicted_label, 'actual:', test[i][0])

    if test[i][0] == predicted_label:
        score += 1

print('score:', score, 'total:', len(train), 'acc:', (score / len(train)) * 100)