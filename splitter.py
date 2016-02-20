from random import shuffle
from pyrsistent import pvector

'''
use this to attain a cross-validation
'''
def cross():
    shuffled = None

    def fn(pipe, idx, total):
        nonlocal shuffled

        if not 'x' in pipe:
            raise Exception('no x')

        # do the shuffling, making a better CV
        # do only once
        if not shuffled:
            x = pipe['x']
            y = pipe['y'] if 'y' in pipe else [None for each in x]
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

        segment = shuffled[start: start+size]
        new_x = pvector(list(map(lambda x: x[0], segment)))
        # print('new_x:', new_x)
        new_y = pvector(list(map(lambda x: x[1], segment)))
        # print('new_y:', new_y )
        new_pipe = pipe.set('x', new_x).set('y', new_y)
        return new_pipe

    return fn