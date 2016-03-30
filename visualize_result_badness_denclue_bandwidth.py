import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import json
from util import get_cmap

datasets = [
    'iris_with_test',
    # 'pendigits',
    # 'yeast_with_test',
    # 'satimage',
    # 'banknote_with_test',
    # 'spam_with_test',
    # 'drd_with_test',
    # 'imagesegment',
    # 'pageblock_with_test',
    # 'statlogsegment_with_test',
    # 'winequality_white_with_test',
    # 'winequality_red_with_test',
]

fig, axes = plt.subplots(3, 4)

for ax, dataset in zip(axes.flatten(), datasets):

    with open('results/badness_denclue_bandwidth-' + dataset + '.json') as file:
        result = json.load(file)

    def plot(ax, sort_fn, name=''):

        data = {
            'acc_kmeans_3':
                list(map(lambda x: x[0] / x[1], result['evaluation_kmeans_3'])),
            'badness_denclue_rot':
                list(map(lambda x: x['md'], result['badness_denclue_weighted_rot'])),
            'badness_denclue_cv_ml':
                list(map(lambda x: x['md'], result['badness_denclue_weighted_cv_ml'])),
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
        ax.plot(x, col['badness_denclue_rot'], 'k', color="red", label='rot')
        ax.plot(x, col['badness_denclue_cv_ml'], 'k', color="blue", label='cv_ml')

        plt.sca(ax)

        title = dataset.replace('_with_test', '')
        plt.title(title)
        plt.xticks(range(cnt), col['names'], rotation=90)

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

plt.subplots_adjust(bottom=0.2)

plt.show()