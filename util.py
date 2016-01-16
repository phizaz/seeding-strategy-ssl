import csv

import numpy


def load_data(file):
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

        result = []
        for row in spamreader:
            if len(row) > 0:
                result.append(row)
        return result

def to_number(data):
    return map(lambda row: map(lambda col: float(col),
                               row),
               data)

def rescale(data):
    dataT = numpy.array(data).transpose()

    def rescale_row(row):
        maximum = max(row)
        minimum = min(row)

        return map(lambda col: (col - minimum) / (maximum - minimum),
                   row)

    dataT = map(rescale_row, dataT)

    dataT = to_list(dataT)
    data = numpy.array(dataT).transpose()
    return data


def to_list(data):
    if hasattr(data, '__iter__') and not isinstance(data, str):
        return list(map(lambda x: to_list(x),
                        data))
    else:
        return data

def is_prime(n):
    if n == 2 or n == 3: return True
    if n < 2 or n%2 == 0: return False
    if n < 9: return True
    if n%3 == 0: return False
    r = int(n**0.5)
    f = 5
    while f <= r:
        if n%f == 0: return False
        if n%(f+2) == 0: return False
        f +=6
    return True