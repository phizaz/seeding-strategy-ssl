import numpy as np
import matplotlib.pyplot as plt
import json

dataset = 'pendigits'
with open('results/badness_on_many_seeding-' + dataset + '.json') as file:
    result = json.load(file)

fig, axes = plt.subplots(ncols=2)

def plot(ax, sort_fn):

    data = {
        'acc_kmeans_1':
            list(map(lambda x: x[0] / x[1], result['evaluation_kmeans_1'])),
        'acc_kmeans_3':
            list(map(lambda x: x[0] / x[1], result['evaluation_kmeans_3'])),
        'badness_l_method_md':
            list(map(lambda x: x['md'], result['badness_l_method'])),
        'badness_l_method_weighted_md':
            list(map(lambda x: x['md'], result['badness_l_method_weighted'])),
        'badness_denclue_md':
            list(map(lambda x: x['md'], result['badness_denclue'])),
        'badness_denclue_weighted_md':
            list(map(lambda x: x['md'], result['badness_denclue_weighted'])),
        'badness_naive_md':
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
    ax.plot(x, col['acc_kmeans_1'], 'k--', label='acc kmeans c*1')
    ax.plot(x, col['acc_kmeans_3'], 'k--', color="blue", label='acc kmeans c*3')
    ax.plot(x, col['badness_l_method_md'], 'k', color="red", label='l_method')
    ax.plot(x, col['badness_l_method_weighted_md'], 'k', color="orange", label='l_method+w')
    ax.plot(x, col['badness_denclue_md'], 'k', color="blue", label='kde')
    ax.plot(x, col['badness_denclue_weighted_md'], 'k', color="cyan", label='kde+w')
    ax.plot(x, col['badness_naive_md'], 'k', label='naive')

    plt.sca(ax)
    plt.xticks(range(cnt), col['names'], rotation=90)

    legend = ax.legend(loc='upper right', shadow=True)
    # Now add the legend with some customizations.
    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize('small')

    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width

plot(axes[0], lambda x: x['acc_kmeans_1'])
plot(axes[1], lambda x: x['acc_kmeans_3'])

plt.subplots_adjust(bottom=0.3)

plt.show()