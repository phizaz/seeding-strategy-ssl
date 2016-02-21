from collections import Counter
from random import shuffle, randint
from pyrsistent import pvector

def label_consensus():
    def fn(inst):
        if not 'prediction' in inst:
            raise Exception('no prediction')
        if not 'y' in inst:
            raise Exception('no y')

        prediction = inst['prediction']
        y = inst['y']
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
    def fn(inst):
        if not 'y' in inst:
            raise Exception('no y')

        y = inst['y']
        seq = [i for i in range(len(y))]
        shuffle(seq)
        select_cnt = int(len(y) * prob)
        selected_ids = seq[: select_cnt]
        new_y = [None for i in range(len(y))]
        for id in selected_ids:
            new_y[id] = y[id]
        return pvector(new_y)

    return fn