from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import l_method

def kmeans(*args, **margs):
    def fn(inst):
        if not 'x' in inst:
            raise Exception('no x')

        x = inst['x']

        kmeans = KMeans(*args, **margs)
        kmeans.fit(x)

        return inst.set('model', kmeans).set('prediction', kmeans.labels_)

    return fn

def knn(*args, **margs):
    def fn(inst):
        if not 'x' in inst:
            raise Exception('no x')
        if not 'y' in inst:
            raise Exception('no y')

        x = inst['x']
        y = inst['y']

        knn = KNeighborsClassifier(*args, **margs)
        knn.fit(x, y)

        return inst.set('model', knn)

    return fn

def linear_regression(*args, **margs):
    def fn(inst):
        if not 'x' in inst:
            raise Exception('no x')
        if not 'y' in inst:
            raise Exception('no y')

        x = inst['x']
        y = inst['y']

        regression = LinearRegression(*args, **margs)
        regression.fit(x, y)
        #
        # print('coef:', regression.coef_)
        # print('intercept:', regression.intercept_)

        return inst.set('model', regression)

    return fn

def agglomerative(*args, **margs):
    def fn(inst):
        if not 'x' in inst:
            raise Exception('no x')

        x = inst['x']

        agg = AgglomerativeClustering(*args, **margs)
        agg.fit(x)

        return inst.set('model', agg).set('prediction', agg.labels_)

    return fn

def agglomerative_l_method():
    def fn(inst):
        if not 'x' in inst:
            raise Exception('no x')

        x = inst['x']

        clusters = l_method.agglomerative_l_method(x)
        return inst.set('prediction', clusters)

    return fn