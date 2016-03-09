import numpy as np
import matplotlib.pyplot as plt
import json

dataset = 'pendigits'
with open('results/badness_on_many_seeding-' + dataset + '.json') as file:
    result = json.load(file)

fig, axes = plt.subplots(ncols=2)

def plot(ax, sort=lambda x: x[0]):
    acc_kmeans_1 = list(map(lambda x: x[0] / x[1], result['evaluation_kmeans_1']))
    acc_kmeans_3 = list(map(lambda x: x[0] / x[1], result['evaluation_kmeans_3']))
    badness_l_method_md = list(map(lambda x: x['md'], result['badness']))
    badness_denclue_md = list(map(lambda x: x['md'], result['badness_denclue']))
    badness_naive_md = list(map(lambda x: x['md'], result['badness_naive']))
    names = result['name']

    seq = list(zip(acc_kmeans_1,
                   acc_kmeans_3,
                   badness_l_method_md,
                   badness_denclue_md,
                   badness_naive_md,
                   names))
    seq.sort(key=sort)

    acc_kmeans_1, acc_kmeans_3, badness_l_method_md, badness_denclue_md, badness_naive_md, names = zip(*seq)

    # Example data
    x = range(len(names))

    ax.plot(x, acc_kmeans_1, 'k--', label='acc kmeans c*1')
    ax.plot(x, acc_kmeans_3, 'k--', color="blue", label='acc kmeans c*3')
    ax.plot(x, badness_l_method_md, 'k', color="red", label='bad l_method')
    ax.plot(x, badness_denclue_md, 'k', color="blue", label='bad kde')
    ax.plot(x, badness_naive_md, 'k', label='bad naive')

    plt.sca(ax)
    plt.xticks(range(len(names)), names, rotation=90)

    legend = ax.legend(loc='center right', shadow=True)
    # Now add the legend with some customizations.
    # The frame is matplotlib.patches.Rectangle instance surrounding the legend.
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize('small')

    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width

plot(axes[0], lambda x: x[0])
plot(axes[1], lambda x: x[1])

plt.subplots_adjust(bottom=0.3)

plt.show()