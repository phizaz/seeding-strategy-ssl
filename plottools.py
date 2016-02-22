from matplotlib import pyplot
import numpy

def scatter2d(x, y):
    print('x:', x)
    print('y:', y)
    print('len(x):', len(x))
    print('len(y):', len(y))
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    # pyplot.figure(1, figsize=(8, 8))

    axScatter = pyplot.axes(rect_scatter)
    # axHistx = pyplot.axes(rect_histx)
    # axHisty = pyplot.axes(rect_histy)

    # no labels
    # axHistx.xaxis.set_major_formatter(nullfmt)
    # axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x, y)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = numpy.max([numpy.max(numpy.fabs(x)), numpy.max(numpy.fabs(y))])
    lim = (int(xymax / binwidth) + 1) * binwidth

    axScatter.set_xlim((-lim, lim))
    axScatter.set_ylim((-lim, lim))

    # bins = numpy.arange(-lim, lim + binwidth, binwidth)
    # axHistx.hist(x, bins=bins)
    # axHisty.hist(y, bins=bins, orientation='horizontal')
    #
    # axHistx.set_xlim(axScatter.get_xlim())
    # axHisty.set_ylim(axScatter.get_ylim())

    pyplot.show()