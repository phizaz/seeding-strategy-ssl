from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

def kmeans(*args, **margs):
    def fn(pipe):
        if not 'x' in pipe:
            raise Exception('no x')

        x = pipe['x']

        kmeans = KMeans(*args, **margs)
        kmeans.fit(x)

        return pipe.set('model', kmeans)

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