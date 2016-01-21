import numpy as np
import numpy.random
import matplotlib.pyplot as plt
from functools import reduce

centroids = [
    (2, 4, 12, 1.0),
    (5, 0, 8, 1.0),
    (2, -3, 45, 1.5),
    (-3, -5, 4, 0.5),
    (-3, 1, 35, 1.5)
]

def generate(centroid, count, size):
    result = []
    for i in range(count * 30):
        x = numpy.random.normal(centroid[0], size)
        y = numpy.random.normal(centroid[1], size) * -1
        result.append((x, y))
    return result

points = reduce(lambda old, x: old | set(generate((x[0], x[1]), x[2], x[3])),
                centroids,
                set())

print(points)

# Generate some test data
x = list(map(lambda x: x[0],
             points))
y = list(map(lambda x: x[1],
             points))

heatmap, xedges, yedges = np.histogram2d(y, x, bins=15)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.imshow(heatmap, extent=extent)
plt.show()