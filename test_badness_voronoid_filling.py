from dataset import *
from pipe import Pipe
from wrapper import *
from badness import *
from ssltools import *

dataset = get_iris()

centroids = denclue(dataset.X, dataset.bandwidth, sample_size=len(dataset.X))
weights = get_centroid_weights(dataset.X, centroids)
pipe = Pipe()\
    .x(dataset.X)\
    .y(dataset.Y)\
    .y_seed(seeding_random(0.1))\
    .connect(stop())
y_seed = pipe['y_seed']
print('y_seed:', y_seed)

seeding = list(map(lambda xy: xy[0],
                   filter(lambda xy: xy[1] is not None,
                          zip(dataset.X, y_seed))))

badness = voronoid_filling(seeding, centroids, weights)

print('badness:', badness)