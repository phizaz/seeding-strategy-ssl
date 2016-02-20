import util
from pipe import Pipe
from wrapper import kmeans, predict, dump

file = './datasets/iris/iris.data'
dataset = util.load_data(file, delimiter=',')
# print('dataset:', dataset)
def remove_label(data):
    return map(lambda x: x[:-1],
               data)
points = remove_label(dataset)
points = util.to_number(points)
points = util.to_list(points)
points = util.rescale(points)

# print('points:', points)

a = Pipe()\
    .x(points)\
    .pipe(kmeans(5))\
    .x(points)\
    .pipe(predict())\
    .pipe(dump('prediction'))


