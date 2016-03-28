from pipe import Pipe
from wrapper import kernel_density_estimation
from pipetools import dump, evaluate, start_timer, stop_timer
from utils import load_x

# file = './datasets/pendigits/pendigits.tra'
file = './datasets/iris/iris.data'

Pipe() \
    .x(load_x(file, delimiter=',')) \
    .connect(start_timer()) \
    .pipe(kernel_density_estimation(bandwidth=0.5)) \
    .connect(stop_timer())\
    .pipe(dump('pdf'))