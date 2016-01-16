# KNN with interating over K to find the best K
import sklearn.neighbors

import util

file = './datasets/iris/iris.data'

data = util.load_data(file)

def remove_label(data):
    return map(lambda x: x[:-1],
               data)

train = remove_label(data)
train = util.to_number(train)
train = util.to_list(train)

def get_label(data):
    return map(lambda x: x[-1],
               data)

label = list(get_label(data))
print('label:', label)

#primes = filter(lambda x: util.is_prime(x),
#                range(2, 100))

# odds performed better
odds = filter(lambda x: x % 2,
              range(1, 100))

for neighbor_cnt in odds:
    print('neighbor_cnt:', neighbor_cnt)
    knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=neighbor_cnt)
    knn.fit(train, label)

    results = map(lambda x: knn.predict([x]),
                  train)

    results = util.to_list(results)

    corrects = 0
    for i, guess in enumerate(results):
        corrects += 1 if guess[0] == label[i] else 0

    print(corrects)