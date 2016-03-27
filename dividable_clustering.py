from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import numpy as np

class DividableClustering:

    def __init__(self):
        self.latest_label = 0
        self.X_by_label = {}

    def get_centroid(self, ind_X):
        X = self.X_of(ind_X)
        centroid = np.zeros(len(X[0]))
        for x in X:
            centroid += x
        centroid /= len(X)
        return centroid

    def get_centroids(self):
        centroid_by_label = {}
        for label, X in self.X_by_label.items():
            centroid_by_label[label] = self.get_centroid(X)
        return centroid_by_label

    def X(self):
        X_pool = []
        for _, ind_X in self.X_by_label.items():
            X_pool += ind_X
        X_pool.sort(key=lambda x: x[1])
        return self.X_of(X_pool)

    def Y(self):
        Y_pool = []
        for label, ind_X in self.X_by_label.items():
            indexes = list(map(lambda x: x[1], ind_X))
            Y_pool += [(label, idx) for i, idx in enumerate(indexes)]
        Y_pool.sort(key=lambda x: x[1])
        return self.X_of(Y_pool)

    def attach_idx(self, X, indexes):
        return list(zip(X, indexes))

    def X_of(self, ind_X):
        return list(map(lambda x: x[0], ind_X))

    def fit(self, X, labels):
        self.latest_label = max(labels) + 1
        ind_X = self.attach_idx(X, range(len(X)))
        for x, label in zip(ind_X, labels):
            if label not in self.X_by_label:
                self.X_by_label[label] = []
            self.X_by_label[label].append(x)

    def get_X(self, label):
        return self.X_of(self.X_by_label[label])

    def get_X_with_idx(self, label):
        return self.X_by_label[label]

    def rename_labels(self, labels):
        renamed = list(map(lambda x: x + self.latest_label, labels))
        return renamed

    def split(self, target, new_labels):
        ind_X = self.X_by_label.pop(target)
        new_labels = self.rename_labels(new_labels)
        self.latest_label = max(new_labels) + 1
        for x, label in zip(ind_X, new_labels):
            if label not in self.X_by_label:
                self.X_by_label[label] = []
            self.X_by_label[label].append(x)

    def relabel(self):
        target = 0
        for label in range(self.latest_label):
            label_found = False
            if label in self.X_by_label:
                label_found = True

                if target != label:
                    self.X_by_label[target] = self.X_by_label.pop(label)

            if label_found:
                target += 1

        self.latest_label = target

    def predict_centroids(self, X):
        labels = sorted(self.X_by_label.keys())
        centroids = [self.get_centroid(self.X_by_label[i]) for i in labels]

        # print('centroids:', sorted(centroids, key=lambda x: x[0]))

        knn = KNeighborsClassifier(1)
        knn.fit(centroids, labels)
        return knn.predict(X)

    def predict_nn(self, X):
        X_train = self.X()
        Y_train = self.Y()

        knn = KNeighborsClassifier(1)
        knn.fit(X_train, Y_train)
        return knn.predict(X)
