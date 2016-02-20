import pvectorc
from pyrsistent import v, pvector, pmap

stack = v(pmap({'a': 10}))
stack = stack.append(v(pmap({'a': 10}), pmap({'a': 20})))
stack = stack.append(v(v(pmap({'a': 10}), pmap({'a': 20})), v(pmap({'a': 10}), pmap({'a': 30}))))

print(stack)

def reduce_fn(list):
    s = 0
    for each in list:
        s += each['a']
    return s

def go_deep(l):
    if not isinstance(l, list) and not isinstance(l, pvectorc.PVector):
        return None
    else:
        result = []
        is_ceil = False
        for i, each in enumerate(l):
            a = go_deep(each)
            if a == None:
                is_ceil = True
                break
            else:
                result.append(a)

        if is_ceil:
            return reduce_fn(l)
        else:
            return result

top = stack[-1]
result = go_deep(top)
print(result)
new_top = list(map(lambda x: x[1].set('merge', result[x[0]]),
                   enumerate(stack[-2])))
print(new_top)
new_stack = stack[:-2].append(new_top)
print(new_stack)