import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import json
from util import get_cmap

dataset = 'imagesegment'
with open('results/goodness_cluster_mocking-' + dataset + '.json') as file:
    result = json.load(file)

fig, axes = plt.subplots(ncols=1)

def plot(ax, sort_fn, name=''):

    data = {
        'acc_kmeans_3':
            list(map(lambda x: x[0] / x[1], result['evaluation_kmeans_3'])),
        'label_correctness_kmeans_3':
            list(map(lambda x: x[0] / x[1], result['label_correctness_kmeans_3'])),
        'goodness_cluster_mocking_nested_split_ward': result['goodness_cluster_mocking_nested_split_ward'],
        'goodness_cluster_mocking_nested_split_average': result['goodness_cluster_mocking_nested_split_average'],
        'goodness_cluster_mocking_nested_split_complete': result['goodness_cluster_mocking_nested_split_complete'],
        'goodness_cluster_mocking_nested_split_single': result['goodness_cluster_mocking_nested_split_single'],
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
    ax.plot(x, col['goodness_cluster_mocking_nested_split_ward'], 'k', color='red', label='kmn')
    ax.plot(x, col['goodness_cluster_mocking_nested_split_average'], 'k', color='orange', label='kmn')
    ax.plot(x, col['goodness_cluster_mocking_nested_split_complete'], 'k', color='yellow', label='kmn')
    ax.plot(x, col['goodness_cluster_mocking_nested_split_single'], 'k', color='magenta', label='kmn')

    plt.sca(ax)
    plt.title(dataset + ' ' + name)
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

# plot(axes[0], lambda x: x['acc_kmeans_1'], 'sort by kmeans 1')
plot(axes, lambda x: x['acc_kmeans_3'], 'sort by kmeans 3')

plt.subplots_adjust(bottom=0.3)

plt.show()