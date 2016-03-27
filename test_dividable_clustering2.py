from dividable_clustering import DividableClustering
from agglomerative_clustering import AgglomerativeClustering
from sklearn.cluster import KMeans
from dataset import *
from sklearn.neighbors import BallTree
from l_method import agglomerative_l_method

dataset = get_iris()

l_method = agglomerative_l_method(dataset.X)

model = DividableClustering()
model.fit(dataset.X, l_method.labels_)

print('labels:', l_method.labels_)

print('predicts:', model.predict(dataset.X))