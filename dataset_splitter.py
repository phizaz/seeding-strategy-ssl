from dataset import *
import numpy as np
from os.path import join
from random import shuffle

'''
split dataset into train and test part
'''

datasets = [
    get_iris(),
    get_yeast(),
    get_letter(),
    get_banknote(),
    get_eeg(),
    get_magic(),
    get_spam(),
    get_auslan(1),
    get_drd(),
    get_imagesegment(),
    get_pageblock(),
    get_statlogsegment(),
    get_winequality('white'),
    get_winequality('red')
]

def split(X, Y, test_ratio=0.3):
    X = list(X)
    Y = list(Y)

    XY = list(zip(X, Y))
    shuffle(XY)

    test_count = int(len(X) * test_ratio)

    # train, test
    return XY[test_count:], XY[:test_count]

def save(path, name, train, test):

    def save_each(path, XY):
        with open(path, 'w') as file:
            for x, y in XY:
                x_str = ','.join(map(str, x))
                file.write(x_str + ',' + y + '\n')

    save_each(join(path, name + '.train'), train)
    save_each(join(path, name + '.test'), test)

def split_dataset(dataset, test_ratio=0.3):
    print('working', dataset.name)
    if dataset.has_testdata():
        print('has test data skipping')
        return

    train, test = split(dataset.X, dataset.Y, test_ratio)
    save(dataset.path, dataset.name, train, test)
    print('finish', dataset.name)

for dataset in datasets:
    if dataset.name == 'iris':
        split_dataset(dataset, 0.2)
    else:
        split_dataset(dataset)