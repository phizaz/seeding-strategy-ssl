import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import json

dataset = 'iris'
with open('results/badness_on_many_seeding-' + dataset + '.json') as file:
    result = json.load(file)

fig, axes = plt.subplots(ncols=2)

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
        RGB color.'''
    color_norm = colors.Normalize(vmin=0, vmax=N - 1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')

    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)

    def static(index):
        return 'rgb'[index]

    return map_index_to_rgb_color
    # return static

def plot(ax, sort_fn):

    data = {
        'acc_kmeans_1':
            list(map(lambda x: x[0] / x[1], result['evaluation_kmeans_1'])),
        'acc_kmeans_3':
            list(map(lambda x: x[0] / x[1], result['evaluation_kmeans_3'])),
        # 'badness_l_method_md':
        #     list(map(lambda x: x['md'], result['badness_l_method'])),
        # 'badness_l_method_weighted_md':
        #     list(map(lambda x: x['md'], result['badness_l_method_weighted'])),
        # 'badness_l_method_weighted_voronoid_filling':
        #     list(map(lambda x: x['voronoid_filling'], result['badness_l_method_weighted'])),
        # 'badness_denclue_md':
        #     list(map(lambda x: x['md'], result['badness_denclue'])),
        # 'badness_denclue_weighted_md':
        #     list(map(lambda x: x['md'], result['badness_denclue_weighted'])),
        # 'badness_denclue_weighted_voronoid_filling':
        #     list(map(lambda x: x['voronoid_filling'], result['badness_denclue_weighted'])),
        # 'badness_hierarchical_voronoid_filling': result['badness_hierarchical_voronoid_filling'],
        # 'badness_majority_voronoid': result['badness_majority_voronoid'],
        # 'badness_kmeans_mocking': result['badness_kmeans_mocking'],
        'badness_kmeans_mocking_nested': result['badness_kmeans_mocking_nested'],
        'badness_naive_md':
            list(map(lambda x: x['md'], result['badness_naive'])),
        'names': result['name'],
    }

    # voronoid_c = result['voronoid_c'][0]
    # voronoid_sigmoid = result['voronoid_sigmoid'][0]
    # kmeans_sigmoid = result['kmeans_sigmoid'][0]

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
    ax.plot(x, col['acc_kmeans_1'], 'k--', label='acc c*1')
    ax.plot(x, col['acc_kmeans_3'], 'k--', color="blue", label='acc c*3')
    # ax.plot(x, col['badness_l_method_md'], 'k', color="red", label='l')
    # ax.plot(x, col['badness_l_method_weighted_md'], 'k', color="orange", label='l+w')
    # ax.plot(x, col['badness_l_method_weighted_voronoid_filling'], 'k', color="yellow", label='l+w+vl')
    # ax.plot(x, col['badness_denclue_md'], 'k', color="blue", label='kde')
    # ax.plot(x, col['badness_denclue_weighted_md'], 'k', color="cyan", label='kde+w')
    # ax.plot(x, col['badness_denclue_weighted_voronoid_filling'], 'k', color="green", label='kde+w+vl')

    # cmap = get_cmap(len(voronoid_c) + 1)
    # print('voronoid_c:', len(voronoid_c))
    # transposed = zip(*col['badness_hierarchical_voronoid_filling'])
    # for i, (badness, c) in enumerate(zip(transposed, voronoid_c)):
    #     ax.plot(x, badness, 'k', color=cmap(i), label='c' + str(round(c, 2)))

    # cmap = get_cmap(len(voronoid_sigmoid) + 1)
    # transposed = zip(*col['badness_hierarchical_voronoid_filling'])
    # for i, (badness, sigmoid) in enumerate(zip(transposed, voronoid_sigmoid)):
    #     ax.plot(x, badness, 'k', color=cmap(i), label='c' + str(round(sigmoid, 2)))

    # cmap = get_cmap(len(kmeans_sigmoid) + 1)
    # transposed = zip(*col['badness_kmeans_mocking'])
    # for i, (badness, sigmoid) in enumerate(zip(transposed, kmeans_sigmoid)):
    #     ax.plot(x, badness, 'k', color=cmap(i), label='c' + str(round(sigmoid, 2)))

    ax.plot(x, col['badness_kmeans_mocking_nested'], 'k', color='red', label='kmn')

    # ax.plot(x, col['badness_majority_voronoid'], 'k', color='red', label='naive')
    ax.plot(x, col['badness_naive_md'], 'k', label='naive')

    plt.sca(ax)
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

plot(axes[0], lambda x: x['acc_kmeans_1'])
plot(axes[1], lambda x: x['acc_kmeans_3'])

plt.subplots_adjust(bottom=0.3)

plt.show()