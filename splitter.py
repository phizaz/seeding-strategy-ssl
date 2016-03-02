from random import shuffle
from pyrsistent import pvector
from util import *

'''
use this to attain a cross-validation
'''
def cross(*add_fields):
    # add fields will be regarded as x, y

    shuffled = None

    def fn(inst, idx, total):
        nonlocal shuffled

        x, y = requires(['x', 'y'], inst)

        # do the shuffling, making a better CV
        # do only once
        if not shuffled:
            x, y = requires(['x', 'y'], inst)
            attachments = requires(add_fields, inst)
            # print('x:', x)
            # print('y:', y)
            data = list(zip(x, y, *attachments))
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
        train_x, train_y, *train_attachments = list(zip(*train))

        test = shuffled[start: start + size]
        test_x, test_y, *test_attachments = list(zip(*test))

        # print('test_x:', test_x)
        # print('test_y:', test_y)

        new_inst = inst\
            .set('x', train_x)\
            .set('y', train_y)\
            .set('x_test', test_x)\
            .set('y_test', test_y)

        # additional fields got its train and test versions as well
        for i, field in enumerate(add_fields):
            new_inst = new_inst\
                .set(field, train_attachments[i])\
                .set(field + '_test', test_attachments[i])

        return new_inst

    return fn