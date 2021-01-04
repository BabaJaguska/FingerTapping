import numpy as np

import Tap


# TODO mozda razdvojiti na SignalExtractor i TapExtractor?

# =============================================================================
# SIGNALS - vraca istu ndarray
# =============================================================================

def get_values(signal):
    result = [signal.gyro1x, signal.gyro1y, signal.gyro1z, signal.gyro1x, signal.gyro1y, signal.gyro1z]
    return result


def get_values_scaled(signal):
    gyro1x, gyro1y, gyro1z = signal.gyro1x, signal.gyro1y, signal.gyro1z
    max_1 = max(signal.gyro1Vec)
    gyro1x, gyro1y, gyro1z = gyro1x / max_1, gyro1y / max_1, gyro1z / max_1
    gyro2x, gyro2y, gyro2z = signal.gyro2x, signal.gyro2y, signal.gyro2z
    max_2 = max(signal.gyro2Vec)
    gyro2x, gyro2y, gyro2z = gyro2x / max_2, gyro2y / max_2, gyro2z / max_2

    result = [gyro1x, gyro1y, gyro1z, gyro2x, gyro2y, gyro2z]
    return result


def get_spherical(signal):
    r1, phi1, theta1, r2, phi2, theta2 = signal.transform_spherical()
    result = [r1, phi1, theta1, r2, phi2, theta2]
    return result


def get_spherical_scaled(signal):
    r1, phi1, theta1, r2, phi2, theta2 = signal.transform_spherical()
    max_1 = max(r1)
    r1 = r1 / max_1
    max_2 = max(r2)
    r2 = r2 / max_2

    result = [r1, phi1, theta1, r2, phi2, theta2]
    return result


def get_amplitude(signal):
    result = [signal.gyro1Vec, signal.gyro2Vec]
    return result


def get_signed_amplitude(signal):
    result = [signal.gyro1VecSign, signal.gyro2VecSign]
    return result


def get_diff_signed_amplitude(signal):
    result = [signal.gyro1VecSign - signal.gyro2VecSign, signal.gyro1VecSign + signal.gyro2VecSign]
    return result


def get_square_amplitude(signal):
    result = [signal.gyro1VecSign * signal.gyro1VecSign, signal.gyro2VecSign * signal.gyro2VecSign]
    return result


def get_amplitude_integral(signal):
    sum1 = calc_integral(signal.gyro1VecSign)
    sum2 = calc_integral(signal.gyro2VecSign)

    result = [sum1, sum2]
    return result


def get_amplitude_no_drift_integral(signal):
    sum1 = calc_no_drift_integral(signal.gyro1VecSign)
    sum2 = calc_no_drift_integral(signal.gyro2VecSign)

    result = [sum1, sum2]
    return result


def get_amplitude_scaled(signal):
    values_1 = signal.gyro1Vec
    max_1 = max(signal.gyro1Vec)
    values_1 = values_1 / max_1
    values_2 = signal.gyro2Vec
    max_2 = max(signal.gyro2Vec)
    values_2 = values_2 / max_2

    result = [values_1, values_2]
    return result


def get_spectrogram(signal):
    result = [signal.spectrogram_i, signal.spectrogram_t]
    return result


def get_all(signal):
    result = []
    concatenate_lists(result, get_values(signal))
    concatenate_lists(result, get_values_scaled(signal))
    concatenate_lists(result, get_spherical(signal))
    concatenate_lists(result, get_spherical_scaled(signal))
    concatenate_lists(result, get_amplitude(signal))
    concatenate_lists(result, get_signed_amplitude(signal))
    concatenate_lists(result, get_diff_signed_amplitude(signal))
    concatenate_lists(result, get_square_amplitude(signal))
    concatenate_lists(result, get_amplitude_integral(signal))
    concatenate_lists(result, get_amplitude_no_drift_integral(signal))
    concatenate_lists(result, get_amplitude_scaled(signal))
    concatenate_lists(result, get_spectrogram(signal))

    return result


# =============================================================================
# TAPS
# =============================================================================


def get_taps(signal, list_of_data):
    taps = []
    cnt = 0
    for data in list_of_data:
        taps_i = get_signal_taps(signal, data)
        if len(taps_i) > 0:
            taps.append(taps_i)
        else:
            cnt = cnt + 1
    result = taps
    return result, cnt


def get_signal_taps(signal, data):
    taps = Tap.get_taps(signal, data)
    taps = concatenate_taps(taps)
    return taps


def get_taps_normalised_len(signal, list_of_data):
    taps = []
    cnt = 0
    for data in list_of_data:
        taps_i = get_signal_taps_normalised_len(signal, data)
        if len(taps_i) > 0:
            taps.append(taps_i)
        else:
            cnt = cnt + 1
    result = taps
    return result, cnt


def get_signal_taps_normalised_len(signal, data):
    taps = Tap.get_taps(signal, data)
    stretch_taps = Tap.stretch_time_taps(taps)
    taps = concatenate_taps(stretch_taps)
    return taps


def get_taps_normalised_max_len(signal, list_of_data):
    taps = []
    cnt = 0
    for data in list_of_data:
        taps_i = get_signal_taps_normalised_max_len(signal, data)
        if len(taps_i) > 0:
            taps.append(taps_i)
        else:
            cnt = cnt + 1
    result = taps
    return result, cnt


def get_signal_taps_normalised_max_len(signal, data):
    taps = Tap.get_taps(signal, data)
    max_len = Tap.tap_max_len(taps)
    crop_taps = Tap.crop_time_taps(taps, max_len)
    taps = concatenate_taps(crop_taps)
    return taps


def get_taps_max_len_normalised(signal, list_of_data):
    taps = []
    cnt = 0
    for data in list_of_data:
        taps_i = get_signal_taps_max_len_normalised(signal, data)
        if len(taps_i) > 0:
            taps.append(taps_i)
        else:
            cnt = cnt + 1
    result = taps
    return result, cnt


def get_signal_taps_max_len_normalised(signal, data):
    taps = Tap.get_taps(signal, data)
    max_len = Tap.tap_max_len(taps)
    crop_taps = Tap.crop_time_taps(taps, max_len)
    stretch_taps = Tap.stretch_time_taps(crop_taps)
    taps = concatenate_taps(stretch_taps)
    return taps


def get_taps_double_stretch(signal, list_of_data):
    taps = []
    cnt = 0
    for data in list_of_data:
        taps_i = get_signal_taps_double_stretch(signal, data)
        if len(taps_i) > 0:
            taps.append(taps_i)
        else:
            cnt = cnt + 1
    result = taps
    return result, cnt


def get_signal_taps_double_stretch(signal, data):
    taps = Tap.get_taps(signal, data)
    stretch_taps = Tap.stretch_time_taps(taps)
    double_stretch_taps = Tap.stretch_val_each_taps(stretch_taps)
    taps = concatenate_taps(double_stretch_taps)
    return taps


def get_taps_no_drift_integral(signal, list_of_data):
    taps = []
    cnt = 0
    for data in list_of_data:
        taps_i = get_signal_no_drift_integral(signal, data)
        if len(taps_i) > 0:
            taps.append(taps_i)
        else:
            cnt = cnt + 1
    result = taps
    return result, cnt


def get_signal_no_drift_integral(signal, data):
    taps = Tap.get_taps(signal, data)
    no_drift_integral_taps = Tap.taps_no_drift_integral(taps)
    taps = concatenate_taps(no_drift_integral_taps)
    return taps


def concatenate_taps(taps):
    if len(taps) > 0:
        if len(taps[0].shape) > 1:  # TODO proveriti da li ovo lepo radi sa visedimenzionim nizovima
            result = np.concatenate(taps, axis=-1)
        else:
            result = np.concatenate(taps)
    else:
        result = ()
    return result


# =============================================================================
# UTIL
# =============================================================================


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


def concatenate_lists(list1, list2):
    for element in list2:
        list1.append(element)
    return
