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

# space = np.linspace(0, 1, 10)
# points = cartesian((space, space))

histories = []

def plot_each(each):
    print('plot:', each)
    summit, history = climb(each)
    return summit, history

start_time = time.time()

with ProcessPoolExecutor() as executor:
    # this pattern is important for using multiprocessing
    executors = []
    for i, each in enumerate(X):
        executors.append(executor.submit(plot_each, each))

    summits = []
    for e in executors:
        summit, history = e.result()
        summits.append(summit)
        path_x, path_y = list(zip(*history))
        plt.scatter(path_x, path_y, c='red', edgecolors='none')

    summit_x, summit_y = list(zip(*summits))
    plt.scatter(summit_x, summit_y, c='green')

end_time = time.time()

print('time elapsed:', end_time - start_time)

plt.show()