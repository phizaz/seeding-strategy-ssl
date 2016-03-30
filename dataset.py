from functools import partial
from bandwidth_selection import BandwidthSelection
from util import *
from pipetools import *
from os.path import join, exists

class Dataset:
    def __init__(self,
                 dataset_name,
                 X, Y,
                 X_test=None,
                 Y_test=None,
                 path=None,
                 train_file=None,
                 test_file=None,
                 bandwidth=None):
        self.name = dataset_name
        self.X = X
        self.Y = Y
        self.X_test = X_test
        self.Y_test = Y_test
        self.path = path
        self.train_file = train_file
        self.test_file = test_file
        self.cluster_cnt = self.count_cluster()
        self.K_for_KNN = self.calculate_K_for_KNN()

        if bandwidth is not None:
            self.bandwidth_type = bandwidth
            self.bandwidth = self.get_bandwidth(type=bandwidth)

    def count_cluster(self):
        counter = Counter(self.Y)
        return len(counter)

    def rescale(self):
        X = self.X
        train_cnt = len(X)
        if self.has_testdata():
            X = np.concatenate((X, self.X_test))
        X = rescale(X)
        self.X = X[:train_cnt]
        self.X_test = X[train_cnt:]
        return self

    def calculate_K_for_KNN(self):
        self.rescale()

        storage_file = './storage/dataset_' + self.name + '.json'

        if not exists(storage_file):
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

    def get_bandwidth(self, type, force=False):
        self.rescale()

        get_bandwidth = {
            'rot': BandwidthSelection.gaussian_distribution,
            'cv_ml': BandwidthSelection.cv_maximum_likelihood,
        }

        storage_file = './storage/dataset_' + self.name + '.json'

        if not exists(storage_file):
            # file not found
            open(storage_file, 'w').close()

        with open(storage_file) as file:
            try:
                data = json.load(file)
            except ValueError:
                data = {}

        field = 'bandwidth_' + type

        if not force and field in data:
            # needed information is in the file
            bandwidth = data[field]
            return bandwidth

        if self.X is not None:
            print('calculating bandwidth (', self.bandwidth_type, ') for', self.name)
            data[field] = get_bandwidth[type](self.X)
            # save to database
            with open(storage_file, 'w') as file:
                json.dump(data, file)

            return data[field]
        else:
            raise Exception('X is not properly configured')

    def has_testdata(self):
        return self.X_test is not None and self.Y_test is not None

'''
original datasets
'''

def get_iris(bandwidth=None):
    path = './datasets/iris'
    file = join(path, 'iris.data')
    return Dataset(
        'iris',
        load_x(file),
        load_y(file),
        path=path,
        train_file=file,
        bandwidth=bandwidth,
    )

def get_yeast(bandwidth=None):
    path = './datasets/yeast'
    file = join(path, 'yeast.data')
    return Dataset(
        'yeast',
        load_x(file, ' ', lambda row: row[1:-1]),
        load_y(file, ' ', lambda row: row[-1]),
        path=path,
        train_file=file,
        bandwidth=bandwidth,
    )

def get_letter(bandwidth=None):
    path = './datasets/letter'
    file = join(path, 'letter-recognition.data')
    return Dataset(
        'letter',
        load_x(file, ',', lambda row: row[1:]),
        load_y(file, ',', lambda row: row[0]),
        path=path,
        train_file=file,
        bandwidth=bandwidth,
    )

def get_pendigits(bandwidth=None):
    path = './datasets/pendigits'
    file = join(path, 'pendigits.tra')
    file_test = join(path, 'pendigits.tes')
    return Dataset(
        'pendigits',
        load_x(file),
        load_y(file),
        load_x(file_test),
        load_y(file_test),
        path=path,
        train_file=file,
        test_file=file_test,
        bandwidth=bandwidth,
    )

def get_satimage(bandwidth=None):
    path = './datasets/satimage'
    file = join(path, 'sat.trn')
    file_test = join(path, 'sat.tst')
    _load_x = partial(load_x, delimiter=' ', remove_label=lambda row: row[:-1])
    _load_y = partial(load_y, delimiter=' ', get_label=lambda row: row[-1])
    return Dataset(
        'satimage',
        _load_x(file),
        _load_y(file),
        _load_x(file_test),
        _load_y(file_test),
        path=path,
        train_file=file,
        test_file=file_test,
        bandwidth=bandwidth,
    )

def get_banknote(bandwidth=None):
    path = './datasets/banknote'
    file = join(path, 'data_banknote_authentication.txt')
    return Dataset(
        'banknote',
        load_x(file, ',', lambda row: row[:-1]),
        load_y(file, ',', lambda row: row[-1]),
        path=path,
        train_file=file,
        bandwidth=bandwidth,
    )

def get_eeg(bandwidth=None):
    path = './datasets/eeg'
    file = join(path, 'EEG Eye State.arff')
    return Dataset(
        'eeg',
        load_x(file, ',', lambda row: row[:-1]),
        load_y(file, ',', lambda row: row[-1]),
        path=path,
        train_file=file,
        bandwidth=bandwidth,
    )

def get_magic(bandwidth=None):
    path = './datasets/magic'
    file = join(path, 'magic04.data')
    return Dataset(
        'magic',
        load_x(file, ',', lambda row: row[:-1]),
        load_y(file, ',', lambda row: row[-1]),
        path=path,
        train_file=file,
        bandwidth=bandwidth,
    )

def get_spam(bandwidth=None):
    path = './datasets/spam'
    file = join(path, 'spambase.data')
    return Dataset(
        'spam',
        load_x(file, ',', lambda row: row[:-1]),
        load_y(file, ',', lambda row: row[-1]),
        path=path,
        train_file=file,
        bandwidth=bandwidth,
    )

def get_auslan(id=1, bandwidth=None):
    path = './datasets/auslan'
    file = join(path, 'tctodd/tctodd' + str(id) + '.txt')
    return Dataset(
        'auslan',
        load_x(file, ' ', lambda row: row[:-1]),
        load_y(file, ' ', lambda row: row[-1]),
        path=path,
        train_file=file,
        bandwidth=bandwidth,
    )

def get_drd(bandwidth=None):
    path = './datasets/drd'
    file = join(path, 'messidor_features.arff')
    return Dataset(
        'drd',
        load_x(file, ',', lambda row: row[:-1]),
        load_y(file, ',', lambda row: row[-1]),
        path=path,
        train_file=file,
        bandwidth=bandwidth,
    )

def get_imagesegment(bandwidth=None):
    path = './datasets/imagesegment'
    file = join(path, 'segmentation.data')
    file_test = join(path, 'segmentation.test')
    return Dataset(
        'imagesegment',
        load_x(file, ',', lambda row: row[1:]),
        load_y(file, ',', lambda row: row[0]),
        load_x(file_test, ',', lambda row: row[1:]),
        load_y(file_test, ',', lambda row: row[0]),
        path=path,
        train_file=file,
        test_file=file_test,
        bandwidth=bandwidth,
    )

def get_pageblock(bandwidth=None):
    path = './datasets/pageblock'
    file = join(path, 'page-blocks.data')
    return Dataset(
        'pageblock',
        load_x(file, ' ', lambda row: row[1:]),
        load_y(file, ' ', lambda row: row[-1]),
        path=path,
        train_file=file,
        bandwidth=bandwidth,
    )

def get_statlogsegment(bandwidth=None):
    path = './datasets/statlogsegment'
    file = join(path, 'segment.dat')
    return Dataset(
        'statlogsegment',
        load_x(file, ' ', lambda row: row[:-1]),
        load_y(file, ' ', lambda row: row[-1]),
        path=path,
        train_file=file,
        bandwidth=bandwidth,
    )

def get_winequality(type='white', bandwidth=None):
    path = './datasets/winequality'
    file = {
        'white': join(path, 'winequality-white.csv'),
        'red': join(path, 'winequality-red.csv'),
    }
    return Dataset(
        'winequality_' + type,
        load_x(file[type], ';', lambda row: row[:-1]),
        load_y(file[type], ';', lambda row: row[-1]),
        path=path,
        train_file=file[type],
        bandwidth=bandwidth,
    )

'''
datasets that have been splitted into .train and .test
'''

def get_iris_with_test(bandwidth=None):
    file = './datasets/iris/iris.train'
    file_test = './datasets/iris/iris.test'
    return Dataset(
            'iris_with_test',
            load_x(file),
            load_y(file),
            load_x(file_test),
            load_y(file_test),
            bandwidth=bandwidth,
    )

def get_yeast_with_test(bandwidth=None):
    file = './datasets/yeast/yeast.train'
    file_test = './datasets/yeast/yeast.test'
    return Dataset(
            'yeast_with_test',
            load_x(file),
            load_y(file),
            load_x(file_test),
            load_y(file_test),
            bandwidth=bandwidth,
    )

def get_letter_with_test(bandwidth=None):
    file = './datasets/letter/letter.train'
    file_test = './datasets/letter/letter.test'
    return Dataset(
            'letter_with_test',
            load_x(file),
            load_y(file),
            load_x(file_test),
            load_y(file_test),
            bandwidth=bandwidth,
    )

def get_banknote_with_test(bandwidth=None):
    file = './datasets/banknote/banknote.train'
    file_test = './datasets/banknote/banknote.test'
    return Dataset(
            'banknote_with_test',
            load_x(file),
            load_y(file),
            load_x(file_test),
            load_y(file_test),
            bandwidth=bandwidth,
    )

def get_eeg_with_test(bandwidth=None):
    file = './datasets/eeg/eeg.train'
    file_test = './datasets/eeg/eeg.test'
    return Dataset(
            'eeg_with_test',
            load_x(file),
            load_y(file),
            load_x(file_test),
            load_y(file_test),
            bandwidth=bandwidth,
    )

def get_magic_with_test(bandwidth=None):
    file = './datasets/magic/magic.train'
    file_test = './datasets/magic/magic.test'
    return Dataset(
            'magic_with_test',
            load_x(file),
            load_y(file),
            load_x(file_test),
            load_y(file_test),
            bandwidth=bandwidth,
    )

def get_spam_with_test(bandwidth=None):
    file = './datasets/spam/spam.train'
    file_test = './datasets/spam/spam.test'
    return Dataset(
            'spam_with_test',
            load_x(file),
            load_y(file),
            load_x(file_test),
            load_y(file_test),
            bandwidth=bandwidth,
    )

def get_auslan_with_test(bandwidth=None):
    file = './datasets/auslan/auslan.train'
    file_test = './datasets/auslan/auslan.test'
    return Dataset(
            'auslan_with_test',
            load_x(file),
            load_y(file),
            load_x(file_test),
            load_y(file_test),
            bandwidth=bandwidth,
    )

def get_drd_with_test(bandwidth=None):
    file = './datasets/drd/drd.train'
    file_test = './datasets/drd/drd.test'
    return Dataset(
            'drd_with_test',
            load_x(file),
            load_y(file),
            load_x(file_test),
            load_y(file_test),
            bandwidth=bandwidth,
    )

def get_pageblock_with_test(bandwidth=None):
    file = './datasets/pageblock/pageblock.train'
    file_test = './datasets/pageblock/pageblock.test'
    return Dataset(
            'pageblock_with_test',
            load_x(file),
            load_y(file),
            load_x(file_test),
            load_y(file_test),
            bandwidth=bandwidth,
    )

def get_statlogsegment_with_test(bandwidth=None):
    file = './datasets/statlogsegment/statlogsegment.train'
    file_test = './datasets/statlogsegment/statlogsegment.test'
    return Dataset(
            'statlogsegment_with_test',
            load_x(file),
            load_y(file),
            load_x(file_test),
            load_y(file_test),
            bandwidth=bandwidth,
    )

def get_winequality_with_test(type='white', bandwidth=None):
    file = './datasets/winequality/winequality_' + type + '.train'
    file_test = './datasets/winequality/winequality_' + type + '.test'
    return Dataset(
            'winequality_' + type + '_with_test',
            load_x(file),
            load_y(file),
            load_x(file_test),
            load_y(file_test),
            bandwidth=bandwidth,
    )