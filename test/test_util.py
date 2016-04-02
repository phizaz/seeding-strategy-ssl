from unittest import TestCase
from util import *
import numpy as np

class Test_load_x(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iris = '../datasets/iris/iris.train'

    def test_wrong_delimiter(self):
        with self.assertRaises(Exception):
            X = load_x(self.iris, delimiter=' ', remove_label=lambda x: x[:-1])

    def test_loading(self):
        X = load_x(self.iris, delimiter=',', remove_label=lambda x: x[:-1])

        self.assertEqual(X.shape, (120, 4))
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(X[0], np.ndarray)

class Test_load_y(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iris = '../datasets/iris/iris.train'

    def test_wrong_delimiter(self):
        with self.assertRaises(Exception):
            Y = load_y(self.iris, delimiter=' ', get_label=lambda x: x[-1])

    def test_loading(self):
        Y = load_y(self.iris, delimiter=',', get_label=lambda x: x[-1])

        self.assertEqual(Y.shape, (120,))
        self.assertIsInstance(Y, np.ndarray)
        for y in Y:
            self.assertIsInstance(y, str)

class Test_load_data(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iris = '../datasets/iris/iris.train'


    def test_wrong_delimiter(self):
        data = load_data(self.iris, delimiter=' ')
        self.assertIsInstance(data, list)
        for row in data:
            self.assertEqual(len(row), 1)
            self.assertIsInstance(row[0], str)

    def test_load(self):
        data = load_data(self.iris, delimiter=',')
        self.assertTrue(len(data) > 0)
        self.assertIsInstance(data, list)

        for row in data:
            self.assertIsInstance(row, list)
            self.assertEqual(len(row), 5)
            for col in row:
                self.assertIsInstance(col, str)

class Test_to_number(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = [
            [1, 2, 3],
            ['0.5', '1.4', '0'],
        ]

    def test_to_number(self):
        new_data = to_number(self.data)
        for row in new_data:
            for col in row:
                self.assertIsInstance(col, float)

    def test_new_array(self):
        new_data = to_number(self.data)
        self.assertNotEqual(self.data, new_data)

class Test_rescale(TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = [
            [1, 2, 3],
            [0, 0, 0],
            [-1, -2, -3],
        ]

    def test_rescale(self):
        new_data = rescale(self.data)
        for row in new_data:
            for col in row:
                self.assertTrue(0 <= col <= 1)

class Test_to_list(TestCase):

    def test_to_list(self):
        l = [['c', 'd'], [1,2, ['b', 'c']]]

        l_itr = map(lambda x: x, l)
        ll = to_list(l_itr)
        self.assertEqual(l, ll)

        l_itr = map(lambda x: map(lambda y: y, x), l)
        ll = to_list(l_itr)
        self.assertEqual(l, ll)

class Test_good_K_for_KNN_with(TestCase):

    def test_good_K(self):
        iris = '../datasets/iris/iris.train'
        X = load_x(iris)
        Y = load_y(iris)
        goodK, acc = good_K_for_KNN(X, Y)
        self.assertIsInstance(goodK, int)
        self.assertIsInstance(acc, float)

class Test_good_K_for_KNN_with_testdata(TestCase):

    def test_good_K(self):
        iris = '../datasets/iris/iris.train'
        iris_test = '../datasets/iris/iris.test'
        X = load_x(iris)
        Y = load_y(iris)
        X_test = load_x(iris_test)
        Y_test = load_y(iris_test)

        goodK, acc = good_K_for_KNN_with_testdata(X, Y, X_test, Y_test)
        self.assertIsInstance(goodK, int)
        self.assertIsInstance(acc, float)

class
