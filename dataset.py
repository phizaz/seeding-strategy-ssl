from dataset_loader import load_url
from util import *
from pipetools import *
from functools import partial
from collections import Counter
import os
import json
from kde import get_bandwidth

class Dataset:
    def __init__(self, dataset_name, X, Y, X_test=None, Y_test=None):
        self.name = dataset_name
        self.X = X
        self.Y = Y
        self.X_test = X_test
        self.Y_test = Y_test
        self.cluster_cnt = self.count_cluster()
        self.K_for_KNN = self.calculate_K_for_KNN()

        # self.bandwidth = self.get_bandwidth()

    def count_cluster(self):
        counter = Counter(self.Y)
        return len(counter)

    def calculate_K_for_KNN(self):
        storage_file = './storage/dataset_' + self.name + '.json'

        if not os.path.exists(storage_file):
            # file not found
            open(storage_file, 'w').close()

        with open(storage_file) as file:
            try:
                data = json.load(file)
            except ValueError:
                data = {}

        if 'K_for_KNN' in data:
            # needed information is in the file
            good_K, acc = data['K_for_KNN']
            return good_K

        if self.X is not None and self.Y is not None:
            if self.X_test is not None and self.Y_test is not None:
                # use test data
                good_K, acc = good_K_for_KNN_with_testdata(self.X, self.Y, self.X_test, self.Y_test)
            else:
                # use cross validate
                good_K, acc = good_K_for_KNN(self.X, self.Y)

            # save to database
            data['K_for_KNN'] = good_K, acc
            with open(storage_file, 'w') as file:
                json.dump(data, file)

            return good_K
        else:
            raise Exception('X and Y are not properly configured')

    def get_bandwidth(self, force=False):
        storage_file = './storage/dataset_' + self.name + '.json'

        if not os.path.exists(storage_file):
            # file not found
            open(storage_file, 'w').close()

        with open(storage_file) as file:
            try:
                data = json.load(file)
            except ValueError:
                data = {}

        if not force and 'bandwidth' in data:
            # needed information is in the file
            bandwidth = data['bandwidth']
            return bandwidth

        if self.X is not None:
            data['bandwidth'] = get_bandwidth(self.X)
            # save to database
            with open(storage_file, 'w') as file:
                json.dump(data, file)

            return data['bandwidth']
        else:
            raise Exception('X is not properly configured')

    def has_testdata(self):
        return self.X_test is not None and self.Y_test is not None

def get_iris():
    file = './datasets/iris/iris.data'
    return Dataset(
        'iris',
        load_x(file),
        load_y(file)
    )

def get_yeast():
    file = './datasets/yeast/yeast.data'
    return Dataset(
        'yeast',
        load_x(file, ' ', lambda row: row[1:-1]),
        load_y(file, ' ', lambda row: row[-1])
    )

def get_letter():
    file = './datasets/letter/letter-recognition.data'
    return Dataset(
        'letter',
        load_x(file, ',', lambda row: row[1:]),
        load_y(file, ',', lambda row: row[0])
    )

def get_pendigits():
    file = './datasets/pendigits/pendigits.tra'
    file_test = './datasets/pendigits/pendigits.tes'
    return Dataset(
        'pendigits',
        load_x(file),
        load_y(file),
        load_x(file_test),
        load_y(file_test)
    )

def get_satimage():
    file = './datasets/satimage/sat.trn'
    file_test = './datasets/satimage/sat.tst'
    _load_x = partial(load_x, delimiter=' ', remove_label=lambda row: row[:-1])
    _load_y = partial(load_y, delimiter=' ', get_label=lambda row: row[-1])
    return Dataset(
        'satimage',
        _load_x(file),
        _load_y(file),
        _load_x(file_test),
        _load_y(file_test)
    )

def get_banknote():
    file = './datasets/banknote/data_banknote_authentication.txt'
    return Dataset(
        'banknote',
        load_x(file, ',', lambda row: row[:-1]),
        load_y(file, ',', lambda row: row[-1])
    )

def get_eeg():
    file = './datasets/eeg/EEG Eye State.arff'
    return Dataset(
        'eeg',
        load_x(file, ',', lambda row: row[:-1]),
        load_y(file, ',', lambda row: row[-1])
    )

def get_bankmarket():
    # should we take only numerical part
    raise Exception('not implemented yet!')

def get_magic():
    file = './datasets/magic/magic04.data'
    return Dataset(
        'magic',
        load_x(file, ',', lambda row: row[:-1]),
        load_y(file, ',', lambda row: row[-1])
    )

def get_spam():
    file = './datasets/spam/spambase.data'
    return Dataset(
        'spam',
        load_x(file, ',', lambda row: row[:-1]),
        load_y(file, ',', lambda row: row[-1])
    )

def get_auslan(id=1):
    file = './datasets/auslan/tctodd/tctodd' + str(id) + '.txt'
    return Dataset(
        'auslan',
        load_x(file, ' ', lambda row: row[:-1]),
        load_y(file, ' ', lambda row: row[-1])
    )

def get_drd():
    file = './datasets/drd/messidor_features.arff'
    return Dataset(
        'drd',
        load_x(file, ',', lambda row: row[:-1]),
        load_y(file, ',', lambda row: row[-1])
    )

def get_imagesegment():
    file = './datasets/imagesegment/segmentation.data'
    file_test = './datasets/imagesegment/segmentation.test'
    return Dataset(
        'imagesegment',
        load_x(file, ',', lambda row: row[1:]),
        load_y(file, ',', lambda row: row[0]),
        load_x(file_test, ',', lambda row: row[1:]),
        load_y(file_test, ',', lambda row: row[0])
    )

def get_pageblock():
    file = './datasets/pageblock/page-blocks.data'
    return Dataset(
        'pageblock',
        load_x(file, ' ', lambda row: row[1:]),
        load_y(file, ' ', lambda row: row[-1])
    )

def get_statlogsegment():
    file = './datasets/statlogsegment/segment.dat'
    return Dataset(
        'statlogsegment',
        load_x(file, ' ', lambda row: row[:-1]),
        load_y(file, ' ', lambda row: row[-1])
    )

def get_winequality(type='white'):
    file = {
        'white': './datasets/winequality/winequality-white.csv',
        'red': './datasets/winequality/winequality-red.csv',
    }
    return Dataset(
        'winequality_' + type,
        load_x(file[type], ';', lambda row: row[:-1]),
        load_y(file[type], ';', lambda row: row[-1])
    )
