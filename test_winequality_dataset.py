from dataset import *
from collections import Counter

print(Counter(get_winequality_with_test('red').Y))

print(Counter(get_winequality_with_test('white').Y))