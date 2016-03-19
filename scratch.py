import math

def series(r, length):
    for i in range(length):
        yield math.pow(1-r, i) * r

def find(length, target=1):
    low = 0
    high = 1

    precision = 1e-14

    while True:
        mid = (low + high) / 2

        # s = list(series(mid, length))
        # print('mid:', mid)
        # print('series:', s)

        sum_series = sum(series(mid, length))
        # print('sum:', sum_series)

        if abs(sum_series - target) < precision:
            return mid
        elif sum_series > target:
            high = mid
        else:
            low = mid

result = find(50, 1)

print('result:', result)