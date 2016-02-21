import util
import time

def load_x(file_path, delimiter = ',', remove_label = lambda x: x[:-1]):
    dataset = util.load_data(file_path, delimiter=',')
    points = map(remove_label, dataset)
    points = util.to_number(points)
    points = util.to_list(points)
    points = util.rescale(points)
    return points

def load_y(file_path, delimiter = ',', get_label = lambda x: x[-1]):
    dataset = util.load_data(file_path, delimiter=',')
    points = map(get_label, dataset)
    points = util.to_list(points)
    return points

def predict():
    def fn(inst):
        if not 'model' in inst:
            raise Exception('no model')
        if not 'x' in inst:
            raise Exception('no x')

        x = inst['x']
        model = inst['model']

        prediction = []
        for point in x:
            prediction_array = model.predict([point])
            prediction.append(prediction_array[0])

        # print('prediction:', prediction)

        return inst.set('prediction', prediction)

    return fn

def evaluate():
    def fn(inst):
        if not 'prediction' in inst:
            raise Exception('no prediction')
        if not 'y' in inst:
            raise Exception('no y')

        prediction = inst['prediction']
        y = inst['y']

        if len(prediction) != len(y):
            raise Exception('len y != prediction')

        correct = 0
        total = len(y)
        for y, Y in zip(y, prediction):
            correct += y == Y

        return inst.set('evaluation', (correct, total))

    return fn

def average(field):
    def fn(inst):
        s = 0
        t = 0
        for pipe in inst:
            e = pipe[field]
            # print('eval:', e)
            s += e[0]
            t += e[1]
        return s / len(inst), t / len(inst)

    return fn

def dump(key):
    def fn(inst):
        if not key in inst:
            raise Exception('no ' + key)

        print(key, ':', inst[key])
        return inst
    return fn

def echo(str):
    def fn(pipe):
        print(str)
        return pipe
    return fn

def copy(a, b):
    def fn(inst):
        if not a in inst:
            raise Exception('no ' + a)
        return inst.set(b, inst[a])
    return fn

def start_timer():
    def fn(pipe):
        start_time = time.time()
        return pipe.attach('start_time', start_time)
    return fn

def stop_timer():
    def fn(pipe):
        start_time = pipe.read('start_time')
        end_time = time.time()
        print('time elapsed:', end_time - start_time)
        return pipe.attach('end_time', end_time)
    return fn