# KNN with interating over K to find the best K
import sklearn.neighbors
import sklearn.cross_validation

import util

#file = './datasets/iris/iris.data'
#file =  './datasets/pendigits/pendigits.tra'
file = './datasets/letter-recognition/letter-recognition.data'
#file = './datasets/satimage/sat.trn'

dataset = util.load_data(file, delimiter=',')

def remove_label(data):
    return map(lambda x: x[1:],
               data)

data = remove_label(dataset)
data = util.to_number(data)
data = util.to_list(data)
data = util.rescale(data)

def get_label(data):
    return map(lambda x: x[0],
               data)

target = get_label(dataset)
target = util.to_list(target)

#goodK = util.goodKforKNN(data, target)
goodK = 1
print('goodK:', goodK)

knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=goodK)
scores = sklearn.cross_validation.cross_val_score(knn, data, target, cv=5, n_jobs=2)

print('scores:', scores)
print('cv_acc:', scores.mean())