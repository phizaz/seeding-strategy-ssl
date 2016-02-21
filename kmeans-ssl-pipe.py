import util
import time
from pipe import Pipe
from wrapper import kmeans, knn
from pipetools import *
from splitter import cross
from collections import Counter

from pyrsistent import v, pvector
from random import shuffle, randint

file = './datasets/iris/iris.data'
points = load_x(file, delimiter=',')
target = load_y(file, delimiter=',')

def label_consensus():
    def fn(pipe):
        if not 'prediction' in pipe:
            raise Exception('no prediction')
        if not 'y' in pipe:
            raise Exception('no y')

        prediction = pipe['prediction']
        y = pipe['y']
        # print('y:', y)
        # print('prediction:', prediction)
        group_labels = [None for each in range(max(prediction) + 1)]
        for i, g in enumerate(prediction):
            label = y[i]
            if label:
                if not group_labels[g]:
                    group_labels[g] = Counter()
                group_labels[g][label] += 1
        # print('group_labels:', group_labels)
        majority = list(map(lambda x: x.most_common(1)[0] if x else None, group_labels))
        # print('majority:', majority)
        new_y = [None for i in range(len(y))]
        for i, g in enumerate(prediction):
            # majority comes in (label, freq) or None
            maj = majority[g]
            if maj:
                # if there is a majority
                # take only the first part
                new_y[i] = maj[0]
        # randomly fill the rest (None)
        for i, each in enumerate(new_y):
            if not each:
                # randomly select one label from another
                # if unfortunate we select None again
                # this's why we put it inside while loop
                while True:
                    r = randint(0, len(new_y) - 1)
                    v = new_y[r]
                    if v:
                        new_y[i] = v
                        break
        return pvector(new_y)

    return fn

def random_select_y(prob):
    def fn(pipe):
        if not 'y' in pipe:
            raise Exception('no y')

        y = pipe['y']
        seq = [i for i in range(len(y))]
        shuffle(seq)
        select_cnt = int(len(y) * prob)
        selected_ids = seq[: select_cnt]
        new_y = [None for i in range(len(y))]
        for id in selected_ids:
            new_y[id] = y[id]
        return pvector(new_y)

    return fn

def kmeans_ssl(clusters, neighbors):
    def fn(pipe):
        p = pipe \
            .split(5) \
                .pipe(kmeans(clusters)) \
                .pipe(predict()) \
                .pipe(copy('y', 'y_bak')) \
                .y(random_select_y(0.1)) \
                .y(label_consensus()) \
                .pipe(knn(neighbors)) \
                .pipe(predict()) \
                .pipe(copy('y_bak', 'y')) \
                .pipe(evaluate()) \
            .merge('evaluation', average('evaluation'))
        return p
    return fn

start_time = time.time()

p = Pipe() \
    .x(points) \
    .connect(start_timer())\
    .y(target) \
    .connect(stop_timer())\
    .split(5, cross()) \
        .connect(kmeans_ssl(clusters=3, neighbors=1)) \
    .merge('evaluation', average('evaluation')) \
    .pipe(dump('evaluation'))

end_time = time.time()

print('time elapsed:', (end_time - start_time))