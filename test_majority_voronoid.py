from dataset import *
from pipe import Pipe
from wrapper import *
from badness import *
from ssltools import *

dataset = get_iris()
pipe = Pipe() \
    .x(dataset.X) \
    .y(dataset.Y) \
    .y_seed(seeding_some(0.1, 1)) \
    .connect(stop())
y_seed = pipe['y_seed']
print('y_seed:', y_seed)

badness_engine = MajorityVoronoid(dataset.X)
badness = badness_engine.run(y_seed)

print('badness:', badness)