from pipe import Pipe
from pipetools import *
from multipipetools import *

def merge_sum():
    def fn(insts):
        return sum(inst['pipe_no'] for inst in insts)
    return fn

result = Pipe().split(3).merge('sum', merge_sum()).connect(stop())

print('result:', result)