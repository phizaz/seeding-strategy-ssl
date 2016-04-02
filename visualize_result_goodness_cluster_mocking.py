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

types = ['_', 'ratio', 'split', 'ratio_width', 'split_width']
avgs = {}
bests = {}

for type in types:
    avgs[type] = 0

for ax, dataset in zip(axes.flatten(), datasets):

    with open('results/goodness_cluster_mocking-' + dataset + '.json') as file:
        result = json.load(file)

    def plot(ax, sort_fn, name=''):

        data = {
            'acc_kmeans_3':
                list(map(lambda x: x[0] / x[1], result['evaluation_kmeans_3'])),
            'label_correctness_kmeans_3':
                list(map(lambda x: x[0] / x[1], result['label_correctness_kmeans_3'])),
            'goodness_cluster_mocking': result['goodness_cluster_mocking'],
            'goodness_cluster_mocking_nested_ratio_complete': result['goodness_cluster_mocking_nested_ratio_complete'],
            'goodness_cluster_mocking_nested_split_ward': result['goodness_cluster_mocking_nested_split_ward'],
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
        ax.plot(x, col['goodness_cluster_mocking'], 'k', color='red', label='kmn')
        ax.plot(x, col['goodness_cluster_mocking_nested_ratio_complete'], 'k', color='orange', label='kmn')
        ax.plot(x, col['goodness_cluster_mocking_nested_split_ward'], 'k', color='blue', label='kmn')

        print('dataset:', dataset)


        a = joint_goodness_penalty(col['label_correctness_kmeans_3'], col['goodness_cluster_mocking'], col['goodness_cluster_mocking'])
        L, H = list(zip(*col['goodness_cluster_mocking_nested_ratio_complete']))
        ratio_w = width_penalty(L, H)
        b = joint_goodness_penalty(col['label_correctness_kmeans_3'], L, H)
        L, H = list(zip(*col['goodness_cluster_mocking_nested_split_ward']))
        split_w = width_penalty(L, H)
        c = joint_goodness_penalty(col['label_correctness_kmeans_3'], L, H)

        if 'ratio_dataset' not in bests:
            bests['ratio_dataset'] = (1, None)

        if 'split_dataset' not in bests:
            bests['split_dataset'] = (1, None)

        best_score, best_dataset = bests['ratio_dataset']
        if b < best_score:
            bests['ratio_dataset'] = (b, dataset)

        best_score, best_dataset = bests['split_dataset']
        if c < best_score:
            bests['split_dataset'] = (c, dataset)


        avgs['_'] += a
        avgs['ratio'] += b
        avgs['split'] += c

        avgs['ratio_width'] += ratio_w
        avgs['split_width'] += split_w


        print('penalty (_):', a)
        print('penalty (ratio):', b)
        print('penalty (split):', c)

        print('width (ratio):', ratio_w)
        print('width (split):', split_w)

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

for type, avg in avgs.items():
    avg /= len(datasets)
    print('avg penalty (' + type + '):', avg)

for type, best in bests.items():
    print('best (' + type + '):', best)

plt.subplots_adjust(bottom=0.1)
plt.show()