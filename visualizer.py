import numpy as np
import matplotlib.pyplot as plt
import util
from matplotlib.ticker import NullFormatter

file = './datasets/iris/iris.data'
data = util.load_data(file)

label = 'Iris-versicolor'
features = [0, 1]

def select_only(label, data):
    return filter(lambda x: x[4] == label,
                  data)

data = select_only(label, data)

def remove_label(data):
    return map(lambda x: x[:-1],
               data)

data = remove_label(data)

data = util.convert_to_number(data)

data = util.to_list(data)

data = util.rescale(data)

# data
x = list(map(lambda x: x[features[0]],
             data))
y = list(map(lambda x: x[features[1]],
             data))

nullfmt = NullFormatter()         # no labels

# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left + width + 0.02

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

# start with a rectangular Figure
plt.figure(1, figsize=(8, 8))

axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

# the scatter plot:
axScatter.scatter(x, y)

# now determine nice limits by hand:
binwidth = 0.25
xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
lim = (int(xymax/binwidth) + 1) * binwidth

axScatter.set_xlim((-lim, lim))
axScatter.set_ylim((-lim, lim))

bins = np.arange(-lim, lim + binwidth, binwidth)
axHistx.hist(x, bins=bins)
axHisty.hist(y, bins=bins, orientation='horizontal')

axHistx.set_xlim(axScatter.get_xlim())
axHisty.set_ylim(axScatter.get_ylim())

plt.show()