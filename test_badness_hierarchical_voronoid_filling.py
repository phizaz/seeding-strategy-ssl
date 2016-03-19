from dataset import *
from pipe import Pipe
from wrapper import *
from badness import *
from ssltools import *

dataset = get_iris()
pipe = Pipe() \
    .x(dataset.X) \
    .y(dataset.Y) \
    .y_seed(seeding_some(0.2, 2)) \
    .connect(stop())
y_seed = pipe['y_seed']
print('y_seed:', y_seed)

seeding = list(map(lambda xy: xy[0],
                   filter(lambda xy: xy[1] is not None,
                          zip(dataset.X, y_seed))))

badness = hierarchical_voronoid_filling(seeding, dataset.X)

print('badness:', badness)