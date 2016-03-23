from random import shuffle
from pyrsistent import pvector
from util import *
from util import requires

'''
splitter, use this to attain a cross-validation
'''
def cross(*add_fields):
    # add fields will be regarded as x, y

    shuffled = None

    def fn(inst, idx, total):
        nonlocal shuffled

        # print('idx:', idx, 'total:', total)

        # do the shuffling, making a better CV
        # do only once
        if idx == 0:
            x, y = requires(['x', 'y'], inst)
            # print('x:', x)
            # print('y:', y)
            # print('add_fields:', add_fields)
            attachments = requires(add_fields, inst)
            # print('attachments:', attachments)
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

        # print('train_y:', train_y)

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


'''
mergers
'''
def average(field):
    def fn(insts):
        s = 0
        t = 0
        for inst in insts:
            e = inst[field]
            # print('eval:', e)
            s += e[0]
            t += e[1]
        return s / len(insts), t / len(insts)

    return fn


def total(field):
    def fn(insts):
        s = 0
        t = 0
        for inst in insts:
            e = inst[field]
            # print('eval:', e)
            s += e[0]
            t += e[1]
        return s, t

    return fn


def group(*fields):
    def fn(insts):
        storage = {}

        for field in fields:
            storage[field] = []

        for inst in insts:
            vals = requires(fields, inst)

            for idx, field in enumerate(fields):
                storage[field].append(vals[idx])

        # print('storage:', storage)
        return storage

    return fn

def flat_group(field):
    def fn(insts):
        result = []
        for inst in insts:
            val = requires(field, inst)
            result.append(val)

        return result

    return fn