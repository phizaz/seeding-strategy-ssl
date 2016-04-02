import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
from util import *
import json
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

for ax, dataset in zip(axes.flatten(), datasets):

    with open('results/badness_on_many_seeding-' + dataset + '.json') as file:
        result = json.load(file)

    def plot(ax, sort_fn, name=''):

        data = {
            'acc_kmeans_3':
                list(map(lambda x: x[0] / x[1], result['evaluation_kmeans_3'])),
            'badness_l_method':
                list(map(lambda x: x['md'], result['badness_l_method'])),
            'badness_denclue':
                list(map(lambda x: x['md'], result['badness_denclue'])),
            'badness_naive':
                list(map(lambda x: x['md'], result['badness_naive'])),
            'names': result['name'],
        }

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
        ax.plot(x, col['badness_l_method'], 'k', color="red", label='l')
        ax.plot(x, col['badness_denclue'], 'k', color="blue", label='kde')
        ax.plot(x, col['badness_naive'], 'k', color='grey', label='naive')

        print('dataset:', dataset)
        a = decreasing_penalty(col['badness_l_method'])
        b = decreasing_penalty(col['badness_denclue'])
        c = decreasing_penalty(col['badness_naive'])
        print('score (l_method):', a)
        print('score (denclue):', b)
        print('score (naive):', c)
        mean = (a + b + c) / 3
        print('score (avg):', mean)

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

plt.subplots_adjust(bottom=0.1)

plt.show()