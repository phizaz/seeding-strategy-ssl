from dataset import get_yeast
from sklearn.neighbors import BallTree
from cache import StorageCache
import numpy as np

'''
there is no such case that the same point but different Y
'''

dataset = get_yeast().rescale()

def seed_cache(file):
    file = 'seeding/' + file
    cache = StorageCache(file)
    y_seed = np.array(cache.get())
    return y_seed

# for num in np.linspace(0.01, 0.1, 10):
Y = seed_cache('yeast_some-7-prob-0.1.json')
# Y = seed_cache('yeast_prob-' + str(num) + '.json')
# print('num:', num)

assert len(dataset.X) == len(Y)

XY = list(zip(dataset.X, Y))
# print('XY:', XY)
XY = sorted(XY, key=lambda xy: list(xy[0]))

def equal(a, b):
    for aa, bb in zip(a, b):
        if abs(aa - bb) > 1e-8:
            return False
    return True

for i, (x, y) in enumerate(XY[:-1]):
    next_x, next_y = XY[i + 1]
    if y is not None and next_y is not None and equal(x, next_x) and y != next_y:
        print('found:', y, next_y, x, next_x)
#
# find = [0.35955056, 0.22988506, 0.41772152, 0.37, 0.,
#         0., 0.52054795, 0.16]
#
# for i, (x, y) in enumerate(XY):
#     if equal(x, find):
#         print('!:', y, x)