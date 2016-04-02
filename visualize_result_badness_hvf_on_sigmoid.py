import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import json
from util import *
from util import get_cmap

datasets = [
    'iris_with_test',
    'pendigits',
    'yeast_with_test',
    'satimage',
    'banknote_with_test',
    'spam_with_test',
    'drd_with_test',
    'imagesegment',
    'pageblock_with_test',
    'statlogsegment_with_test',
    'winequality_white_with_test',
    'winequality_red_with_test',
]

fig, axes = plt.subplots(3, 4)

avgs = [0 for i in range(7)]

for ax, dataset in zip(axes.flatten(), datasets):

    with open('results/badness_hvf_on_sigmoid-' + dataset + '.json') as file:
        result = json.load(file)

    def plot(ax, sort_fn, name=''):

        data = {
            'acc_kmeans_3':
                list(map(lambda x: x[0] / x[1], result['evaluation_kmeans_3'])),
            'badness_naive':
                list(map(lambda x: x['md'], result['badness_naive'])),
            'badness_hierarchical_voronoid_filling': result['badness_hierarchical_voronoid_filling'],
            'voronoid_sigmoid': result['voronoid_sigmoid'],
            'names': result['name'],
        }

        # sigmoids are same for all rows
        sigmoids = data['voronoid_sigmoid'][0]

        # transpose the dictionary
        keys = data.keys()
        seq = []
        for values in zip(*data.values()):
            seq.append(dict(zip(keys, values)))

        # sort by sort_fn
        seq.sort(key=sort_fn)

        # transpose it back!
        def dict_to_list(d, order):
            l = [d[o] for o in order]
            return l
        sorted_data_wo_keys = list(map(lambda d: dict_to_list(d, keys),
                                       seq))
        sorted_data = dict(zip(keys, zip(*sorted_data_wo_keys)))

        # Example data
        cnt = len(sorted_data['names'])
        x = range(cnt)

        col = sorted_data
        ax.plot(x, col['acc_kmeans_3'], 'k--', color="black", label='acc c*3')
        ax.plot(x, col['badness_naive'], 'k', color='grey', label='naive')

        hvf = result['badness_hierarchical_voronoid_filling']
        hvf_by_sigmoid = [[] for i in range(len(sigmoids))]
        for each in hvf:
            for i, s in enumerate(each):
                hvf_by_sigmoid[i].append(s)

        cmap = get_cmap(len(sigmoids) + 1)

        print('dataset:', dataset)
        for i, (sigmoid, each_hvf) in enumerate(zip(sigmoids, hvf_by_sigmoid)):
            ax.plot(x, each_hvf, 'k', color=cmap(i), label=sigmoid)
            penalty = decreasing_penalty(each_hvf)
            if i == 2:
                # the best now
                print('penalty:', penalty)
            avgs[i] += penalty


        # remove y axis
        ax.yaxis.set_major_formatter(plt.NullFormatter())

        # scale y to [0,1]
        ax.set_ylim([0, 1])

        plt.sca(ax)
        title = dataset.replace('_with_test', '')
        plt.title(title)

        # increase space between rows
        plt.subplots_adjust(hspace=.5)

        # rename the xticks
        col_names = []
        for col_name in col['names']:
            if col_name.startswith('some'):
                col_name = col_name.replace('some-', '')
                col_name, *_ = col_name.split('-prob')
            elif col_name.startswith('prob'):
                col_name = col_name.replace('prob-', '')
            col_names.append(col_name)
        plt.xticks(range(cnt), col_names, rotation=90)

        # legend = ax.legend(loc='lower center', shadow=True)
        # # Now add the legend with some customizations.
        # # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
        # frame = legend.get_frame()
        # frame.set_facecolor('0.90')
        #
        # # Set the fontsize
        # for label in legend.get_texts():
        #     label.set_fontsize('small')
        #
        # for label in legend.get_lines():
        #     label.set_linewidth(1)  # the legend line width


    plot(ax, lambda x: x['acc_kmeans_3'], 'sort by kmeans 3')

for i in range(len(avgs)):
    avgs[i] /= len(datasets)
    print('penalty (' + str(i) + '):', avgs[i])

plt.subplots_adjust(bottom=0.1)

plt.show()