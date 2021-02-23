import numpy as np


def calc_integral(data):
    result = np.zeros(data.shape)
    prev = 0
    num = data.shape[0]
    for i in range(num):
        result[i] = prev + data[i]
        prev = result[i]
    return result


def calc_linear_drift(data):
    start_drift = 0
    end_drift = data[-1]
    num = data.shape[0]
    delta = (end_drift - start_drift) / num
    drift = np.zeros(data.shape)
    prev = 0
    for i in range(num):
        drift[i] = prev + delta
        prev = drift[i]
    return drift


def calc_no_drift_integral(data):
    integral = calc_integral(data)
    drift = calc_linear_drift(integral)

    result = integral - drift
    return result


def calc_diff(data):
    result = np.zeros(data.shape)
    num = data.shape[0]
    for i in range(num - 1):
        result[i + 1] = data[i + 1] - data[i]
    return result


def concatenate_lists(list1, list2):
    for element in list2:
        list1.append(element)
    return
