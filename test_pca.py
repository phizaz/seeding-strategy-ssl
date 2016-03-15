from sklearn.decomposition import PCA
from dataset import *
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
from cache import StorageCache

dataset = get_iris()
# print('bandwidth:', dataset.get_bandwidth(force=True))

pca = PCA(n_components=2)
pca.fit(dataset.X)
X = pca.transform(dataset.X)
Y = dataset.Y
print('covariance:', pca.get_covariance())

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


cmap = get_cmap(dataset.cluster_cnt + 1)
print('cluster_cnt:', dataset.cluster_cnt)

def plot(X, **kwargs):
    x, y = list(zip(*X))
    plt.scatter(x, y, **kwargs)

group = {}
for x, y in zip(X, Y):
    if y not in group:
        group[y] = []

    group[y].append(x)

for i, (name, points) in zip(range(dataset.cluster_cnt), group.items()):
    # plot X on it deverse it using color according to different Y
    print('color:', cmap(i))
    print('count:', len(points))
    plot(points, color=cmap(i))

cache = StorageCache('seeding/' + dataset.name + '_some-1-prob-0.1.json')
seeds = cache.get()
seeds = list(map(lambda x: x[1],
                 filter(lambda x: x[0],
                        zip(seeds, dataset.X))))
seeds = pca.transform(seeds)
plot(seeds, color='grey')

cache = StorageCache('storage/centroids_iris_denclue_bandwidth_0.1082972972972973.json')
centroids = np.array(cache.get())
centroids = pca.transform(centroids)
plot(centroids, color='black')

plt.show()