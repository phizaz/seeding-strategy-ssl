import time

from util import *


def load_x(file_path, delimiter = ',', remove_label = lambda x: x[:-1]):
    dataset = load_data(file_path, delimiter=delimiter)
    points = list(map(remove_label, dataset))
    points = to_number(points)
    points = to_list(points)
    points = rescale(points)
    return points

def load_y(file_path, delimiter = ',', get_label = lambda x: x[-1]):
    dataset = load_data(file_path, delimiter=delimiter)
    points = map(get_label, dataset)
    points = to_list(points)
    return points

def assign(field, fn):
    def fn(inst):
        return inst.set(field, fn(inst))

    return fn

def predict():
    def fn(inst):
        if not 'model' in inst:
            raise Exception('no model')
        if not 'x_test' in inst:
            raise Exception('no x_test')

        x_test = inst['x_test']
        model = inst['model']

        prediction = model.predict(x_test)

        # print('prediction:', prediction)

        return inst.set('prediction', prediction)

    return fn

def evaluate():
    def fn(inst):
        if not 'prediction' in inst:
            raise Exception('no prediction')
        if not 'y_test' in inst:
            raise Exception('no y_test')

        prediction = inst['prediction']
        y_test = inst['y_test']

        if len(prediction) != len(y_test):
            raise Exception('len y_test != prediction')

        correct = 0
        total = len(y_test)
        for y, Y in zip(y_test, prediction):
            correct += y == Y

        return inst.set('evaluation', (correct, total))

    return fn

def label_correctness(field_y_pred, field_y_ori):
    def fn(inst):
        y_pred, y_ori = requires([field_y_pred, field_y_ori], inst)

        assert len(y_ori) == len(y_pred)

        correct = 0
        for pred, ori in zip(y_pred, y_ori):
            correct += pred == ori

        correctness = correct, len(y_ori)
        return inst.set('label_correctness', correctness)

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

def let(field, val):
    def fn(inst):
        return inst.set(field, val)
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

def stop():
    # get result left from the pipe and stop it
    def fn(pipe):
        if len(pipe.stack) > 1:
            raise Exception('trying to stop many pipes at the same time')

        # no more pipe
        return pipe.stack[0]
    return fn