from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier

def kmeans(*args, **margs):
    def fn(pipe):
        if not 'x' in pipe:
            raise Exception('no x')

        x = pipe['x']

        kmeans = KMeans(*args, **margs)
        kmeans.fit(x)

        return pipe.set('model', kmeans).set('prediction', kmeans.labels_)

    return fn

def knn(*args, **margs):
    def fn(pipe):
        if not 'x' in pipe:
            raise Exception('no x')
        if not 'y' in pipe:
            raise Exception('no y')

        x = pipe['x']
        y = pipe['y']

        knn = KNeighborsClassifier(*args, **margs)
        knn.fit(x, y)

        return pipe.set('model', knn)

    return fn

def agglomerative(*args, **margs):
    def fn(pipe):
        if not 'x' in pipe:
            raise Exception('no x')

        x = pipe['x']

        agg = AgglomerativeClustering(*args, **margs)
        agg.fit(x)

        return pipe.set('model', agg).set('prediction', agg.labels_)

    return fn