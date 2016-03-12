from dataset import *
from util import *
from pyrsistent import pvector

dataset = get_iris()
dump_array_to_file(pvector(dataset.Y), 'test.json')

result = read_file_to_array('test.json')
print('result:', result)
