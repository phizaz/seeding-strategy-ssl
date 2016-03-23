from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

class DividableKmeans:

    def __init__(self):
        self.cluster_centers_ = []
        self.knn = KNeighborsClassifier(1)
        self.X_by_label = {}

    def fit(self, X, cluster_cnt):
        model = KMeans(cluster_cnt)
        model.fit(X)

        self.cluster_centers_ += model.cluster_centers_

        labels = model.predict(X)

        for x, label in zip(X, labels):
            if label not in self.X_by_label:
                self.X_by_label[label] = []
            self.X_by_label[label].append(x)

    def split(self, label, cluster_cnt):

        pass

    def predict(self, X):
        self.knn.fit(self.cluster_centers_)
        return self.knn.predict(X)