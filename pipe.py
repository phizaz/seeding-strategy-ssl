import pvectorc

from pyrsistent import pmap, pvector, v, m

class Pipe:

    def __init__(self, stack = None):
        if not stack:
            stack = v(m())
        self.stack = stack

    def split(self, count, map_fn = None):
        # map_fn(x, idx, total) -> map(x)

        # [A]
        # [A, [B, C]]
        # [A, [B, C], [[C, D, E], [F, G, H]]]

        def default_map(x, *args):
            return x

        # if map function is not given, use default_map
        # which returns exactly the same thing as input (clone)
        if not map_fn:
            map_fn = default_map

        top = self.stack[-1]
        def multiplier(l, cnt):
            if not isinstance(l, pvectorc.PVector):
                return pvector(list(map(lambda x: map_fn(x[1], x[0], cnt),
                                        enumerate([l] * cnt))))
            else:
                return pvector(list(map(lambda x: multiplier(x, cnt), l)))

        multiplied = multiplier(top, count)
        new_stack = self.stack.append(multiplied)
        return Pipe(new_stack)

    def merge(self, key, reduce_fn):
        # reduce_fn(list of pipe instance) -> any

        def go_deep(l):
            if not isinstance(l, pvectorc.PVector):
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
                    return pvector(result)

        result = go_deep(self.stack[-1])
        #print('result:', result)
        second_top = self.stack[-2]
        #print('second_top:', second_top)
        if not isinstance(second_top, pvectorc.PVector):
            new_top = second_top.set(key, result)
        else:
            new_top = pvector(list(map(lambda x: x[1].set(key, result[x[0]]),
                                   enumerate(second_top))))
        #print(new_top)
        new_stack = self.stack[:-2].append(new_top)
        return Pipe(new_stack)

    def _traverse(self, fn):
        # fn(instance)
        def run(l):
            if not isinstance(l, pvectorc.PVector):
                return fn(l)
            else:
                return pvector(list(map(lambda x: run(x), l)))
        new_stack = run(self.stack)
        return new_stack

    '''
    assign onto 'x', input can be either value or function to be executed
    '''
    def x(self, input):
        if hasattr(input, '__call__'):
            new_stack = self._traverse(lambda x: x.set('x', input(x)))
        else:
            new_stack = self._traverse(lambda x: x.set('x', input))

        return Pipe(new_stack)

    '''
    assign onto 'y', input can be either value or function to be executed
    '''
    def y(self, input):
        if hasattr(input, '__call__'):
            new_stack = self._traverse(lambda x: x.set('y', input(x)))
        else:
            new_stack = self._traverse(lambda x: x.set('y', input))

        return Pipe(new_stack)

    def pipe(self, fn):
        # fn(instance) -> instance
        new_stack = self._traverse(fn)
        return Pipe(new_stack)
