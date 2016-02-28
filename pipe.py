import pvectorc
from parmap import parmap
from pyrsistent import pmap, pvector, v, m

# cpu_count = multiprocessing.cpu_count()
cpu_count = 2

print('cpu:', cpu_count)

class Pipe:

    def __init__(self, pipe = None, new_stack = None):
        if not pipe:
            stack = v(m())
            dump = m()
        else:
            stack = pipe.stack
            dump = pipe.dump

        if new_stack:
            stack = new_stack

        self.stack = stack
        self.dump = dump

    def attach(self, key, val):
        self.dump = self.dump.set(key, val)
        return self

    def read(self, key):
        if not key in self.dump:
            raise Exception('no ' + key + ' in pipe\'s dump')
        return self.dump[key]

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

        def multiplier(l, cnt):
            if not isinstance(l, pvectorc.PVector):
                return pvector(list(map(lambda x: map_fn(x[1], x[0], cnt),
                                        enumerate([l] * cnt))))
            else:
                return pvector(list(map(lambda x: multiplier(x, cnt), l)))

        top = self.stack[-1]
        multiplied = multiplier(top, count)
        new_stack = self.stack.append(multiplied)
        # print('new_stack:', new_stack)
        return Pipe(self, new_stack)

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
        return Pipe(self, new_stack)

    '''
    traverse, updatede to ues multi-core processors
    '''
    def _traverse(self, fn):
        def probe(l):
            if not isinstance(l, pvectorc.PVector):
                return [l]
            else:
                result = []
                for x in l:
                    tmp = probe(x)
                    result += tmp
                return result

        # print('stack:', self.stack)
        top = self.stack[-1]
        # print('top:', top)
        queue = probe(top)
        # print('queue:', queue)

        # multicore parallel map
        result = parmap(fn, queue, cpu_count)
        # singlecore map
        # result = list(map(fn, queue))

        # print('result:', result)
        result_pt = 0

        def run(l):
            if not isinstance(l, pvectorc.PVector):
                nonlocal result_pt
                result_pt += 1
                return result[result_pt - 1]
            else:
                return pvector(list(map(lambda x: run(x), l)))

        new_stack = self.stack[:-1].append(run(top))
        return new_stack

    '''
    assign onto 'x', input can be either value or function to be executed
    '''
    def x(self, input):
        if hasattr(input, '__call__'):
            new_stack = self._traverse(lambda x: x.set('x', input(x)))
        else:
            new_stack = self._traverse(lambda x: x.set('x', input))

        return Pipe(self, new_stack)

    '''
    assign onto 'y', input can be either value or function to be executed
    '''
    def y(self, input):
        if hasattr(input, '__call__'):
            new_stack = self._traverse(lambda x: x.set('y', input(x)))
        else:
            new_stack = self._traverse(lambda x: x.set('y', input))

        return Pipe(self, new_stack)

    def pipe(self, fn):
        # fn(instance) -> instance
        new_stack = self._traverse(fn)
        return Pipe(self, new_stack)

    '''
    connect is for chaining an arbitrary function to the pipe (not mapping function)
    '''
    def connect(self, fn):
        return fn(self)
