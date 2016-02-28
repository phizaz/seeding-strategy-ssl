from pipe import Pipe
from wrapper import recursive_agglomerative_l_method, agglomerative_l_method
from pipetools import *
from plottools import scatter2d
import matplotlib.pyplot as plt

# file = './datasets/pendigits/pendigits.tra'
file = './datasets/iris/iris.data'

result = Pipe() \
    .x(load_x(file, delimiter=',')) \
    .connect(start_timer())\
    .pipe(agglomerative_l_method())\
    .connect(stop_timer())\
    .connect(stop())

X = load_x(file)
centroids = result['centroids']

x = list(map(lambda x: x[0], X))
y = list(map(lambda x: x[1], X))

cen_x = list(map(lambda x: x[0], centroids))
cen_y = list(map(lambda x: x[1], centroids))

plt.scatter(x, y, c='blue')
plt.scatter(cen_x, cen_y, c='red')
plt.grid(True)
plt.show()