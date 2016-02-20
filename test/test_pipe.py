from pipe import Pipe

def sumsum(list):
    #print(list)
    s = 0
    for each in list:
        s += each['i'] * 2
    #print('s:', s)
    return s

a = Pipe()\
    .x([1,2,3,4])\
    .y([0, 0, 0, 1])\
    .split(2, lambda x, i, t: x) \
        .split(2, lambda x, i, t: x.set('i', i)) \
        .merge('sum', sumsum)\
    .merge('avg', lambda list: sum(x['sum'] for x in list) / len(list))

print(a.stack)