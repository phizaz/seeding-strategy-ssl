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

def convert_to_number(data):
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
    if hasattr(data, '__iter__'):
        return list(map(lambda x: to_list(x),
                        data))
    else:
        return data

