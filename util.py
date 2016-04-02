import numpy
import numpy as np
import random
from collections import Counter
from operator import itemgetter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
import json
from sklearn.neighbors import BallTree
import matplotlib.cm as cmx
import matplotlib.colors as colors
import pyrsistent

def load_x(file_path, delimiter = ',', remove_label = lambda x: x[:-1]):
    '''
    load file content (XY) to an ndarray by removing the label (Y) part
    also, converting content into float based
    :param file_path: str
    :param delimiter: str
    :param remove_label: function (list) -> list without Y
    :return: ndarray
    '''
    dataset = load_data(file_path, delimiter=delimiter)
    assert len(dataset) > 0
    assert len(dataset[0]) > 1

    points = list(map(remove_label, dataset))
    points = to_number(points)
    points = to_list(points)
    return np.array(points)


def load_y(file_path, delimiter = ',', get_label = lambda x: x[-1]):
    '''
    load file content (XY) to an ndarray by removing the data (X) part
    :param file_path: str
    :param delimiter: str
    :param get_label: function (list) -> Y idx to be preserved
    :return: ndarray
    '''
    dataset = load_data(file_path, delimiter=delimiter)
    assert len(dataset) > 0
    assert len(dataset[0]) > 1

    points = map(get_label, dataset)
    points = to_list(points)
    return np.array(points)

def load_data(file, delimiter=','):
    '''
    load data from file convert to list
    :param file: str
    :param delimiter: str
    :return: list (of list)
    '''
    result = []
    for line in open(file):
        line = line.strip()
        if len(line) == 0:
            continue

        row = line.split(delimiter)
        row = list(filter(lambda x: len(x) > 0, row))
        result.append(row)

    # print('result:', result)
    return result

def to_number(data):
    '''
    convert data (string based) into float based
    :param data: list of list or ndarray
    :return: iterable of iterable
    '''
    return map(lambda row: map(float, row),
               data)

def rescale(data):
    '''
    rescale data to be in range [0, 1]
    :param data: ndarray, list
    :return: ndarary
    '''
    # this shall not be invoked separately !!!
    dataT = numpy.array(data).transpose()

    def rescale_row(row):
        maximum = max(row)
        minimum = min(row)

        if abs(maximum - minimum) < 1e-8:
            # zero variance
            return map(lambda col: 0.0, row)

        return map(lambda col: (col - minimum) / (maximum - minimum),
                   row)

    dataT = map(rescale_row, dataT)

    dataT = to_list(dataT)
    data = numpy.array(dataT).transpose()
    return data


def to_list(data):
    """
    recursively turn iterative data into list
    :param data: iterable
    :return: list
    """
    if hasattr(data, '__iter__') and not isinstance(data, str):
        return list(map(lambda x: to_list(x),
                        data))
    else:
        return data


def array_to_list(array):
    assert isinstance(array, np.ndarray)
    return array.tolist()
    # if array is None or isinstance(array, (str, int, float)):
    #     return array
    # else:
    #     result = []
    #     for row in array:
    #         result.append(array_to_list(row))
    #
    #     # print('result:', result)
    #     return result

def dump_array_to_file(input, file):
    #print('dumping:', input)

    with open(file, 'w') as file:
        json.dump(array_to_list(input), file)

def read_file_to_array(file):
    def arrayization(object):
        return np.array(object)

    with open(file) as file:
        return arrayization(json.load(file))

def is_prime(n):
    raise Exception('not use anymore')
    # if n == 2 or n == 3: return True
    # if n < 2 or n%2 == 0: return False
    # if n < 9: return True
    # if n%3 == 0: return False
    # r = int(n**0.5)
    # f = 5
    # while f <= r:
    #     if n%f == 0: return False
    #     if n%(f+2) == 0: return False
    #     f +=6
    # return True

def random_list(n):
    raise Exception('use random.shuffle instead')
    # l = [i for i in range(n)]
    # for i in range(n):
    #     rand = random.randint(0, n-1)
    #     if rand != i:
    #         l[i], l[rand] = l[rand], l[i]
    # return l

def most_common(cluster):
    raise Exception('use Conuter().most_common(1).pop()[0] instead')
    # count = Counter(cluster).most_common()
    # max_count = count[0][1]
    # most_commons = list(filter(lambda x: x[1] == max_count,
    #                            count))
    # result = most_commons[random.randint(0, len(most_commons) - 1)]
    # return result

def good_K_for_KNN_with_testdata(X, Y, X_test, Y_test):
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)
    assert isinstance(X_test, np.ndarray)
    assert isinstance(Y_test, np.ndarray)

    accuracies = []
    descending_cnt = 0
    descending_threshold = 15

    for k in range(1, len(X) + 1):
        # print('k:', k)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X, Y)
        Y_pred = knn.predict(X_test)
        score = sum(np.array_equal(a, b) for a, b in zip(Y_pred, Y_test))
        score /= len(X)
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


def good_K_for_KNN(X, Y):
    assert isinstance(X, np.ndarray)
    assert isinstance(Y, np.ndarray)

    accuracies = []
    descending_cnt = 0
    descending_threshold = 15

    for k in range(1, len(X) + 1):
        # print('k:', k)
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, Y, cv=5, n_jobs=2)
        score = scores.mean()
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

def requires(request, d):
    assert isinstance(d, (dict, pyrsistent._pmap.PMap))

    if isinstance(request, str):
        if request not in d:
            raise Exception('no ' + request)
        return d[request]

    result = []
    for field in request:
        if field not in d:
            raise Exception('no ' + field)
        result.append(d[field])

    return result

def get_centroid_weights(X, centroids):
    assert isinstance(X, np.ndarray)
    assert isinstance(centroids, np.ndarray)

    ball_tree = BallTree(centroids)
    dist, indexes = ball_tree.query(X)
    weights = [0 for i in centroids]
    for idx in indexes:
        weights[idx] += 1

    return weights

def get_cmap(N):
    '''
    Returns a function that maps each index in 0, 1, ... N-1 to a distinct
        RGB color.
    :param N: int
    :return: function (i) -> ith-color
    '''
    assert isinstance(N, int)

    color_norm = colors.Normalize(vmin=0, vmax=N - 1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')

    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)

    def static(index):
        return 'rgb'[index]

    return map_index_to_rgb_color

def decreasing_penalty(l):
    assert isinstance(l, (list, tuple))

    if len(l) == 1:
        return 0

    score = 0
    for i in range(1, len(l)):
        score += max(0, l[i] - l[i-1])

    return score / (max(l) - min(l))

def goodness_penalty(X, L, H):
    raise Exception('not in use anymore')
    # score = 0
    # for x, l, h in zip(X, L, H):
    #     score += abs(x - l) + abs(x - h)
    # scaled = score / (2 * len(X))
    # return scaled

def width_penalty(L, H):
    assert isinstance(L, (list, tuple))
    assert isinstance(H, (list, tuple))
    assert len(L) == len(H)

    s = sum(h - l for h, l in zip(H, L))
    return s / len(L)

def outrange_rate(X, L, H):
    assert isinstance(X, (list, tuple))
    assert isinstance(L, (list, tuple))
    assert isinstance(H, (list, tuple))
    assert len(X) == len(L) == len(H)

    score = 0
    for x, l, h in zip(X, L, H):
        if x < l or h < x:
            score += 1
    return score / len(X)

def joint_goodness_penalty(X, L, H, C=0.5):
    assert isinstance(X, (list, tuple))
    assert isinstance(L, (list, tuple))
    assert isinstance(H, (list, tuple))
    assert len(X) == len(L) == len(H)

    width = width_penalty(L, H)
    outrange = outrange_rate(X, L, H)
    return C * width + (1 - C) * outrange

def average_width(L, H):
    raise Exception('use width penalty instead')
    # assert isinstance(L, list)
    # assert isinstance(H, list)
    # assert len(L) == len(H)
    #
    # s = sum(h - l for h, l in zip(H, L))
    # return s / len(L)