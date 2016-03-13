from pipe import Pipe
from dataset import *
from ssltools import *
from pipetools import *
from wrapper import *

dataset = get_iris()

result = Pipe()\
    .x(dataset.X)\
    .y(dataset.Y)\
    .x_test(dataset.X)\
    .y_test(dataset.Y)\
    .y_seed(seeding_random(.01))\
    .pipe(kmeans(3))\
    .y(label_consensus())\
    .pipe(knn(1))\
    .pipe(predict())\
    .pipe(evaluate())\
    .connect(stop())

print('result:', result['evaluation'])

