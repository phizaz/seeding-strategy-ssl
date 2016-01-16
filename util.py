import csv


def load_data(file):
    with open(file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')

        result = []
        for row in spamreader:
            result.append(row)
        return result

def to_list(data):
    if hasattr(data, '__iter__'):
        return list(map(lambda x: to_list(x),
                        data))
    else:
        return data

