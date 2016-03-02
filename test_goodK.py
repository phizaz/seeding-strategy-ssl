from pipe import *
from wrapper import *
from pipetools import *

file = './datasets/pendigits/pendigits.tra'
file_test = './datasets/pendigits/pendigits.tes'

X = load_x(file)
Y = load_y(file)
X_test = load_x(file_test)
Y_test = load_y(file_test)

goodK = Pipe()\
    .x(X)\
    .y(Y)\
    .x_test(X_test)\
    .y_test(Y_test)\
    .pipe(good_K_for_KNN())\
    .connect(stop())

print('goodK:', good_K_for_KNN)