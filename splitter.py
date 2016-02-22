from random import shuffle
from pyrsistent import pvector

'''
use this to attain a cross-validation
'''
def cross():
    shuffled = None

    def fn(inst, idx, total):
        nonlocal shuffled

        if not 'x' in inst:
            raise Exception('no x')

        # do the shuffling, making a better CV
        # do only once
        if not shuffled:
            x = inst['x']
            y = inst['y'] if 'y' in inst else [None for each in x]
            # print('x:', x)
            # print('y:', y)
            data = list(zip(x, y))
            # print(data)
            # return
            shuffle(data)
            shuffled = data

        avg_size = int(len(shuffled) / total)
        start = avg_size * idx
        size = avg_size if idx < total - 1 else len(shuffled) - start
        # print('idx:', idx)
        # print('start:', start)
        # print('size:', size)
        train = shuffled[:start] + shuffled[start + size:]
        train_x, train_y = list(zip(*train))

        test = shuffled[start: start + size]
        test_x, test_y = list(zip(*test))

        # print('test_x:', test_x)
        # print('test_y:', test_y)

        new_inst = inst\
            .set('x', train_x)\
            .set('y', train_y)\
            .set('x_test', test_x)\
            .set('y_test', test_y)

        return new_inst

    return fn