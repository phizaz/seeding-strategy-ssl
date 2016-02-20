import pvectorc
from pyrsistent import v

def multiplier(l, cnt):
    if not isinstance(l, list) and not isinstance(l, pvectorc.PVector):
        return list(map(lambda x: x[1],
                        enumerate([l] * cnt)))
    else:
        return list(map(lambda x: multiplier(x, cnt), l))

a = {}
b = multiplier(a, 2)
print(b)
c = multiplier(b, 3)
print(c)