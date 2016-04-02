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

types = ['single', 'complete', 'average', 'ward']
avgs = {}

for ax, dataset in zip(axes.flatten(), datasets):

    with open('results/goodness_cluster_mocking_ratio-' + dataset + '.json') as file:
        result = json.load(file)

    def plot(ax, sort_fn, name=''):

        data = {
            'acc_kmeans_3':
                list(map(lambda x: x[0] / x[1], result['evaluation_kmeans_3'])),
            'label_correctness_kmeans_3':
                list(map(lambda x: x[0] / x[1], result['label_correctness_kmeans_3'])),
            'goodness_cluster_mocking_nested_ratio_ward': result['goodness_cluster_mocking_nested_ratio_ward'],
            'goodness_cluster_mocking_nested_ratio_average': result['goodness_cluster_mocking_nested_ratio_average'],
            'goodness_cluster_mocking_nested_ratio_complete': result['goodness_cluster_mocking_nested_ratio_complete'],
            'goodness_cluster_mocking_nested_ratio_single': result['goodness_cluster_mocking_nested_ratio_single'],
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
        ax.plot(x, col['label_correctness_kmeans_3'], 'k--', color="grey", label='label c*3')
        ax.plot(x, col['goodness_cluster_mocking_nested_ratio_ward'], 'k', color='red', label='kmn')
        ax.plot(x, col['goodness_cluster_mocking_nested_ratio_average'], 'k', color='orange', label='kmn')
        ax.plot(x, col['goodness_cluster_mocking_nested_ratio_complete'], 'k', color='yellow', label='kmn')
        ax.plot(x, col['goodness_cluster_mocking_nested_ratio_single'], 'k', color='magenta', label='kmn')

        print('dataset:', dataset)
        for type in types:
            acc = col['label_correctness_kmeans_3']
            L, H = list(zip(*col['goodness_cluster_mocking_nested_ratio_' + type]))
            penalty = joint_goodness_penalty(acc, L, H)
            print('penalty (' + type + '):', penalty)
            if type not in avgs:
                avgs[type] = 0
            avgs[type] += penalty

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

    # plot(axes[0], lambda x: x['acc_kmeans_1'], 'sort by kmeans 1')
    plot(ax, lambda x: x['acc_kmeans_3'], 'sort by kmeans 3')

for type in types:
    avgs[type] /= len(datasets)
    print('avg penalty (' + type + '):', avgs[type])

plt.subplots_adjust(bottom=0.1)
plt.show()