import numpy as np
from scipy.signal import butter, filtfilt
from numpy.polynomial import Chebyshev


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


def calc_no_drift_integral_filter(data, time_tap=[]):
    integral = calc_integral(data)
    lowcut = 0.4
    highcut = 50
    nf = 200 #[Hz]
    order = 5
    low, high = lowcut/nf, highcut/nf
    b, a = butter(order, [low, high], btype = 'band')
    result = filtfilt(b, a, integral, method = 'gust')
    return result

def calc_no_drift_integral_poly(data, time_tap):
    # NE MOZE NA TAP DA SE PRIMENI JEDAN
    result = []
    integral = calc_integral(data)
    time_tap2 = [t - time_tap[0] for t in time_tap]
    time_tap2[-1] = time_tap2[-1] - 1
    
    y_tap = integral[time_tap2]
    
    # plt.plot(integral[:1000],'k'), plt.stem(time_tap[:10], integral[time_tap[:10]], 'r', use_line_collection=True), plt.show()
    deg = 5
    cc = Chebyshev.fit(time_tap2, y_tap, deg)
    xx, yy = cc.linspace(len(integral))
    result = integral - yy
    
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
