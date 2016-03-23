from dividable_clustering import DividableClustering
from agglomerative_clustering import AgglomerativeClustering
from sklearn.cluster import KMeans
from dataset import *
from sklearn.neighbors import BallTree

dataset = get_iris()

agg = AgglomerativeClustering(3)
agg.fit(dataset.X)

model = DividableClustering()
model.fit(dataset.X, agg.labels_)

print(len(model.X_by_label[0]))
print(len(model.X_by_label[1]))
print(len(model.X_by_label[2]))

kmeans = KMeans(3)
kmeans.fit(model.get_X(0))

model.split(0, kmeans.labels_)

print(len(model.X_by_label[3]))
print(len(model.X_by_label[4]))
print(len(model.X_by_label[5]))

print(model.X_by_label.keys())

model.relabel()

print(model.X_by_label.keys())
print(len(model.X_by_label[0]))
print(len(model.X_by_label[1]))
print(len(model.X_by_label[2]))
print(len(model.X_by_label[3]))
print(len(model.X_by_label[4]))

result = model.predict(dataset.X)
print(result)

for i, r in enumerate(result):
    if r == 3:
        print(i)

ball_tree = BallTree(dataset.X)
dist, _ = ball_tree.query([dataset.X[50], dataset.X[100]], 2)
print(dist)

dist, _ = ball_tree.query(dataset.X, 2)
avg = sum(d[1] for d in dist) / len(dist)
print('avg:', avg)