from kde import *
from dataset import *
from numpy import array
import numpy as np
from cartesian import cartesian
from concurrent.futures import ProcessPoolExecutor

iris = get_iris()

X = list(map(lambda x: x[:2], iris.X))
climb = create_hill_climber(X, .001)

x, y = list(zip(*X))
plt.scatter(x, y, c='blue')

space = np.linspace(0, 1, 5)
points = cartesian((space, space))

histories = []

def plot_each(each):
    print('plot:', each)
    summit, history = climb(each)
    return history

start_time = time.time()

with ProcessPoolExecutor() as executor:
    # this pattern is important for using multiprocessing
    executors = []
    for i, each in enumerate(points):
        executors.append(executor.submit(plot_each, each))

    for e in executors:
        histories.append(e.result())


end_time = time.time()

print('time elapsed:', end_time - start_time)

print('plotting..')

for history in histories:
    path_x, path_y = list(zip(*history))
    plt.scatter(path_x, path_y, c='red', edgecolors='none')

plt.show()