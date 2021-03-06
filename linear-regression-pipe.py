from plottools import scatter2d
import numpy
from collections import deque

x = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101]
# x = list(map(lambda x: [x], x))
y = [69.188980996061389, 58.118232735023263, 45.933845620517651, 36.783575738032425, 34.838679375093101, 29.393325500484156, 22.452663580899817, 22.173786998883905, 21.558839987857532, 20.27786084100444, 18.420163545925067, 17.826167026536801, 15.974445525163004, 15.459729760389521, 14.18903525896604, 11.848941687875698, 11.663618653579171, 11.435679060735575, 11.423237923769554, 11.184903441084343, 10.62391742870682, 10.503270669893167, 10.069861399799482, 10.044782223711486, 9.8045787219253686, 9.4393409143320124, 9.0950336148989077, 8.577392171691649, 8.5500210517787174, 8.4722597981045382, 7.8861146415909156, 7.5007814586794481, 7.4886187663515438, 7.3328646549727692, 6.9887510578441701, 6.887855968362202, 6.7432667991956086, 6.544140825015349, 6.4076980691949954, 6.0758992233086788, 6.0554991116773715, 6.019280381982246, 5.8595773420597634, 5.8556608644875938, 5.751897503026731, 5.7247864333607472, 5.7191231176772472, 5.6746769492816558, 5.6262813341555997, 5.3679408669802227, 5.3633407866221958, 5.0034401057518076, 4.9651282704166277, 4.8857305837318554, 4.8744734341557781, 4.7728318525530877, 4.7707826616781599, 4.7405253965185095, 4.6332970075181024, 4.6180088334900891, 4.5541842347132118, 4.53446478807333, 4.4280085293942397, 4.4079901439245752, 4.3910651579582236, 4.3870783619322395, 4.3384434582699809, 4.322989477868366, 4.310551421851919, 4.300822284073603, 4.2911667127860103, 4.1282152675850092, 4.1233132886111816, 4.0250567644173607, 4.0187329046907641, 3.9896912020525028, 3.9781282041694039, 3.9408400891970943, 3.9287482692281221, 3.9085324459328596, 3.8681949980712513, 3.8336818358519018, 3.7223795781039386, 3.6686774291534565, 3.6561206002051168, 3.5676623625862822, 3.5509914127347662, 3.5396381569402515, 3.5026513097701564, 3.4653969773143825, 3.451806313320616, 3.4302813633700744, 3.4198119811422756, 3.3852262036067904, 3.3569882534535171, 3.3387703623441269, 3.3217175647955735, 3.2610494403459991, 3.2482043695719063, 3.1981526025875531]

x = x[:100]
y = y[:100]

def f_creator(coef, intercept):
    def f(x):
        return intercept + coef * x

    return f


def best_fit_line(x, y):
    # regression = LinearRegression()
    # regression.fit(x, y)
    # coef = regression.coef_[0]
    # intercept = regression.intercept_

    coef, intercept = numpy.polyfit(x, y, 1)
    return coef, intercept


def mean_squared_error(X, Y):
    # start_time = time.time()
    coef, intercept = best_fit_line(X, Y)
    # end_time = time.time()
    # print('best_fit time:', end_time - start_time)

    f = f_creator(coef, intercept)

    sum = 0
    for arr_x, real_y in zip(X, Y):
        x = arr_x
        y = f(x)
        sum += (real_y - y) ** 2
    mse = sum / len(Y)
    return mse


def l_method(num_groups, merge_dist):
    # print(num_groups)
    # print(merge_dist)

    b = len(num_groups) + 1

    # start_time = time.time()
    x_left = num_groups[:2]
    y_left = merge_dist[:2]
    # we use 'deque' data structure here to attain the efficient 'popleft'
    x_right = deque(num_groups[2:])
    y_right = deque(merge_dist[2:])
    # end_time = time.time()
    # print('list preparation time:', end_time - start_time)

    min_score = float('inf')
    min_c = None
    for c in range(3, b - 2):
        # start_time = time.time()
        mseA = mean_squared_error(x_left, y_left)
        mseB = mean_squared_error(x_right, y_right)
        # end_time = time.time()

        # if c % 13 == 0:
        #     print('c:', c)
        #     print('mean_squared_time:', end_time - start_time)
        A = (c - 1) / (b - 1) * mseA
        B = (b - c) / (b - 1) * mseB
        score = A + B

        if score < min_score:
            # print('score:', score)
            # print('c:', c)
            # print('A:', A)
            # print('B:', B)
            # print('mseA:', mseA)
            # print('mseB:', mseB)
            min_c, min_score = c, score

        # start_time = time.time()
        x_left.append(num_groups[c - 1])
        y_left.append(merge_dist[c - 1])

        x_right.popleft()
        y_right.popleft()
        # end_time = time.time()
        # print('list manipulation time:', end_time - start_time)

    return min_c


def refined_l_method(num_groups, merge_dist):
    cutoff = last_knee = current_knee = len(num_groups)
    while True:
        last_knee = current_knee
        print('cutoff:', cutoff)
        current_knee = l_method(num_groups[:cutoff], merge_dist[:cutoff])
        print('current_knee:', current_knee)
        cutoff = current_knee * 3
        if current_knee >= last_knee:
            break
    return current_knee

min_c = refined_l_method(x, y)

print('min_c:', min_c)
scatter2d(x, y)
