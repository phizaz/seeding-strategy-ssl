from dataset import *
from tmp_kmeans_split1 import KmeansMockingNestedSplit as KmeansSplit1
from tmp_kmeans_split2 import KmeansMockingNestedSplit as KmeansSplit2
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from cache import StorageCache
from sklearn.cluster import KMeans

data1 = get_iris()
data2 = get_iris()

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

def seed_cache(file):
    file = 'seeding/' + file
    cache = StorageCache(file)
    y_seed = np.array(cache.get())
    return y_seed

def sync_label(A, B):
    map_A = {}
    map_B = {}
    for a, b in zip(A, B):
        if a not in map_A:
            map_A[a] = b
        if b not in map_B:
            map_B[b] = a
        if map_A[a] != b or map_B[b] != a:
            return False

    AA = list(map(lambda a: map_A[a], A))
    return AA, B

random_cache = seed_cache('iris_prob-0.06.json')

kmeans = KMeans(data1.cluster_cnt * 3)
kmeans.fit(data1.X)

engine1 = KmeansSplit1(data1.cluster_cnt * 3, data1.X, kmeans.labels_)
engine1_goodness = engine1.run(random_cache)
engine2 = KmeansSplit2(data1.cluster_cnt * 3, data1.X, kmeans.labels_)
engine2_goodness = engine2.run(random_cache)

print('engine1_goodness:', engine1_goodness)
print('engine2_goodness:', engine2_goodness)

engine1.groups.sort(key=lambda g: len(g.X))
engine2.groups.sort(key=lambda g: len(g.X))

for group1, group2 in zip(engine1.groups, engine2.groups):
    seeding_cnt1 = group1.seeding_cnt()
    seeding_cnt2 = group2.seeding_cnt()

    print('seeding_cnt:', seeding_cnt1, seeding_cnt2)

    if seeding_cnt1 == 0 and seeding_cnt2 == 0:
        continue

    group1_labels = group1.clustering_model.Y()
    group2_labels = group2.clustering_model.Y()

    sync = sync_label(group1_labels, group2_labels)
    print('sync:', sync)

    print('labels cnt:', group1.clustering_model.latest_label, group2.clustering_model.latest_label)

    print(group1_labels)
    print(group2_labels)

    # seeds = group1.seeds()
    # seed_x, seed_y = list(map(list, zip(*seeds)))
    #
    # model = group1.clustering_model
    # label_cnt = model.latest_label
    # print('label_cnt:', label_cnt)
    #
    # cmap = get_cmap(label_cnt + 1)
    #
    # group_sizes = [(label, len(model.get_X(label))) for label in range(label_cnt)]
    # largest_group, _ = max(group_sizes, key=lambda x: x[1])
    #
    # print('largest_group:', largest_group, 'cnt:', len(model.get_X(largest_group)))
    #
    # centroid = model.get_centroid(model.get_X_with_idx(largest_group))
    # print('centroid:', centroid)
    #
    # pca = PCA(2)
    # pca.fit(group1.X)
    #
    # pca_centroid = pca.transform([centroid])[0]
    # print('pca_centroid:', pca_centroid)
    #
    # c_x = pca_centroid[0]
    # c_y = pca_centroid[1]
    #
    # def plot(X, **kwargs):
    #     X = pca.transform(X)
    #     x1, x2 = list(zip(*X))
    #     plt.scatter(x1, x2, **kwargs)
    #
    # for label in range(label_cnt):
    #     X = model.get_X(label)
    #     print('label:', label, 'cnt:', len(X))
    #     plot(X, c=cmap(label), edgecolors='none')
    #
    # plot(seed_x, c='black')
    # # plt.xlim(-1 + c_x, 1 + c_x)
    # # plt.ylim(-1 + c_y, 1 + c_y)
    #
    # plt.show()