import util
import numpy

a = [1,2,3]
b = util.to_list(a)

print(b)

c = map(lambda x: x,
        a)

d = util.to_list(c)
print(d)

e = numpy.array([['setosa'],['setosa'],['setosa']])
f = map(lambda x: x[-1],
        e)
g = util.to_list(f)
print(g)
