from dataset import *
from util import *

dataset = get_iris()
dump_array_to_file(dataset.X, 'test.json')

result = read_file_to_array('test.json')
print('result:', result)
