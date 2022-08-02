import random
import numpy


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def Dsigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def number_2_array(num, length):
    ret = []
    for i in range(length):
        if i == num:
            ret.append(1)
        else:
            ret.append(0)
    return ret


def array_2_number(array):
    max_list = []
    max_num = 0
    for i, x in enumerate(array):
        if x == max_num:
            max_list.append(i)
        if x > max_num:
            max_num = x
            max_list = [i]
    return max_list[random.randint(0, len(max_list) - 1)]


def get_value():
    value = 0
    negative = bool(random.randint(0, 1))
    while value in (0.0, 1.0):
        value = round(random.random(), 1)
    if negative:
        value = -1 * value
    return value


def custom_sigmoid(x):
    return sigmoid(x - 1)
