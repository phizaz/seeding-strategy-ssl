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

def predict():
    def fn(pipe):
        if not 'model' in pipe:
            raise Exception('no model')
        if not 'x' in pipe:
            raise Exception('no x')

        x = pipe['x']
        model = pipe['model']

        prediction = []
        for point in x:
            prediction_array = model.predict([point])
            prediction.append(prediction_array[0])

        # print('prediction:', prediction)

        return pipe.set('prediction', prediction)

    return fn

def evaluation():
    def fn(pipe):
        if not 'prediction' in pipe:
            raise Exception('no prediction')
        if not 'y' in pipe:
            raise Exception('no y')

        prediction = pipe['prediction']
        y = pipe['y']

        if len(prediction) != len(y):
            raise Exception('len y != prediction')

        correct = 0
        total = len(y)
        for y, Y in zip(y, prediction):
            correct += y == Y

        return pipe.set('evaluation', (correct, total))

    return fn

def average(field):
    def fn(list):
        s = 0
        t = 0
        for pipe in list:
            e = pipe[field]
            # print('eval:', e)
            s += e[0]
            t += e[1]
        return s, t

    return fn

def dump(key):
    def fn(pipe):
        if not key in pipe:
            raise Exception('no ' + key)

        print(key, ':', pipe[key])
        return pipe
    return fn

def copy(a, b):
    def fn(pipe):
        if not a in pipe:
            raise Exception('no ' + a)
        return pipe.set(b, pipe[a])
    return fn