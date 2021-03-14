import numpy as np
from scipy import signal

import Parameters
import Tap
import Util


# TODO mozda razdvojiti na SignalExtractor, TapExtractor, ConvolutionExtractor?

# =============================================================================
# SIGNALS - vraca listu signala (list, ndarray)
# =============================================================================

def get_one_signal(measurement):  # TODO obrisati ovo je za test
    result = [measurement.gyro1x]
    return result


def get_values(measurement):
    result = np.array([measurement.gyro1x, measurement.gyro1y, measurement.gyro1z, measurement.gyro1x, measurement.gyro1y,
              measurement.gyro1z])
    return result


def get_values_scaled(measurement):
    gyro1x, gyro1y, gyro1z = measurement.gyro1x, measurement.gyro1y, measurement.gyro1z
    max_1 = max(measurement.gyro1Vec)
    gyro1x, gyro1y, gyro1z = gyro1x / max_1, gyro1y / max_1, gyro1z / max_1
    gyro2x, gyro2y, gyro2z = measurement.gyro2x, measurement.gyro2y, measurement.gyro2z
    max_2 = max(measurement.gyro2Vec)
    gyro2x, gyro2y, gyro2z = gyro2x / max_2, gyro2y / max_2, gyro2z / max_2

    result = [gyro1x, gyro1y, gyro1z, gyro2x, gyro2y, gyro2z]
    return result


def get_spherical(measurement):
    r1, phi1, theta1, r2, phi2, theta2 = measurement.transform_spherical()
    result = [r1, phi1, theta1, r2, phi2, theta2]
    return result


def get_spherical_scaled(measurement):
    r1, phi1, theta1, r2, phi2, theta2 = measurement.transform_spherical()
    max_1 = max(r1)
    r1 = r1 / max_1
    max_2 = max(r2)
    r2 = r2 / max_2

    result = [r1, phi1, theta1, r2, phi2, theta2]
    return result


def get_amplitude(measurement):
    result = [measurement.gyro1Vec, measurement.gyro2Vec]
    return result


def get_signed_amplitude(measurement):
    result = [measurement.gyro1VecSign, measurement.gyro2VecSign]
    return result


def get_diff_signed_amplitude(measurement):
    result = [measurement.gyro1VecSign - measurement.gyro2VecSign, measurement.gyro1VecSign + measurement.gyro2VecSign]
    return result


def get_square_amplitude(measurement):
    result = [measurement.gyro1VecSign * measurement.gyro1VecSign, measurement.gyro2VecSign * measurement.gyro2VecSign]
    return result


def get_amplitude_integral(measurement):
    sum1 = Util.calc_integral(measurement.gyro1VecSign)
    sum2 = Util.calc_integral(measurement.gyro2VecSign)

    result = [sum1, sum2]
    return result


def get_amplitude_no_drift_integral(measurement):
    sum1 = Util.calc_no_drift_integral(measurement.gyro1VecSign)
    sum2 = Util.calc_no_drift_integral(measurement.gyro2VecSign)

    result = [sum1, sum2]
    return result


def get_amplitude_scaled(measurement):
    values_1 = measurement.gyro1Vec
    max_1 = max(measurement.gyro1Vec)
    values_1 = values_1 / max_1
    values_2 = measurement.gyro2Vec
    max_2 = max(measurement.gyro2Vec)
    values_2 = values_2 / max_2

    result = [values_1, values_2]
    return result


def get_spectrogram(measurement):  # TODO mozda podeliti spektrograme ako gornja i dolja polovina nose iste podatke
    result = [measurement.spectrogram_i, measurement.spectrogram_t]
    return result


def get_max_spectrogram(measurement):
    def max_indexes(spectrogram):
        result_indexes = np.ndarray(shape=(spectrogram.shape[1],))
        for i in range(spectrogram.shape[1]):
            max_val = 0
            max_index = 0
            for j in range(spectrogram.shape[0]):
                a = spectrogram[j][i]
                if a > max_val:
                    max_val = a
                    max_index = j
            result_indexes[i] = max_val
        # TODO maksimum je uvek na sredini, osim prvih nekoliko koji su svi nule, tako da nema smisla pratiti indeks vec vrednosti sto se sada radi
        return result_indexes

    result = [max_indexes(measurement.spectrogram_i), max_indexes(measurement.spectrogram_t)]
    return result


def get_all(measurement):
    result = []
    Util.concatenate_lists(result, get_values(measurement))
    Util.concatenate_lists(result, get_values_scaled(measurement))
    Util.concatenate_lists(result, get_spherical(measurement))
    Util.concatenate_lists(result, get_spherical_scaled(measurement))
    Util.concatenate_lists(result, get_amplitude(measurement))
    Util.concatenate_lists(result, get_signed_amplitude(measurement))
    Util.concatenate_lists(result, get_diff_signed_amplitude(measurement))
    Util.concatenate_lists(result, get_square_amplitude(measurement))
    Util.concatenate_lists(result, get_amplitude_integral(measurement))
    Util.concatenate_lists(result, get_amplitude_no_drift_integral(measurement))
    Util.concatenate_lists(result, get_amplitude_scaled(measurement))
    Util.concatenate_lists(result, get_spectrogram(measurement))
    Util.concatenate_lists(result, get_max_spectrogram(measurement))

    return result


def list_to_array(list_of_signals):
    result = []
    for signal in list_of_signals:
        if len(signal.shape) > 1:
            result.append(signal)
        else:
            result.append([signal])
    result = np.concatenate(result) if len(result) > 0 else []
    return result


# =============================================================================
# TAPS - vraca listu ndarray i int
# =============================================================================

def get_tap(measurement, signals):
    result = [np.asarray(signals)]
    return result


def get_taps(measurement, signals):
    result = get_taps_for_function(measurement, signals, get_signal_taps)
    return result


def get_signal_taps(measurement, signal):
    taps = Tap.get_signal_taps(measurement, signal)
    return taps


def get_taps_normalised_len(measurement, signals):
    result = get_taps_for_function(measurement, signals, get_signal_taps_normalised_len)
    return result


def get_signal_taps_normalised_len(measurement, signal):
    taps = Tap.get_signal_taps(measurement, signal)
    stretch_taps = Tap.stretch_time_taps(taps)
    return stretch_taps


def get_taps_normalised_max_len(measurement, signals):
    result = get_taps_for_function(measurement, signals, get_signal_taps_normalised_max_len)
    return result


def get_signal_taps_normalised_max_len(measurement, signal):
    taps = Tap.get_signal_taps(measurement, signal)
    max_len = Tap.tap_max_len(taps)
    crop_taps = Tap.crop_signal_time_taps(taps, max_len)
    return crop_taps


def get_taps_max_len_normalised(measurement, signals):
    result = get_taps_for_function(measurement, signals, get_signal_taps_max_len_normalised)
    return result


def get_signal_taps_max_len_normalised(measurement, signal):
    taps = Tap.get_signal_taps(measurement, signal)
    max_len = Tap.tap_max_len(taps)
    crop_taps = Tap.crop_signal_time_taps(taps, max_len)
    stretch_taps = Tap.stretch_time_taps(crop_taps)
    return stretch_taps


def get_taps_set_len(measurement, signals):
    result = get_taps_for_function(measurement, signals, get_signal_taps_set_len)
    return result


def get_signal_taps_set_len(measurement, signal, max_len=Parameters.max_tap_len):
    taps = Tap.get_signal_taps(measurement, signal)
    crop_taps = Tap.crop_signal_time_taps(taps, max_len)
    stretch_taps = Tap.stretch_time_taps(crop_taps)
    return stretch_taps


def get_taps_double_stretch(measurement, signals):
    result = get_taps_for_function(measurement, signals, get_signal_taps_double_stretch)
    return result


def get_signal_taps_double_stretch(measurement, signal):
    taps = Tap.get_signal_taps(measurement, signal)
    stretch_taps = Tap.stretch_time_taps(taps)
    double_stretch_taps = Tap.stretch_val_each_taps(stretch_taps)
    return double_stretch_taps


def get_taps_no_drift_integral(measurement, signals):
    result = get_taps_for_function(measurement, signals, get_signal_no_drift_integral)
    return result


def get_signal_no_drift_integral(measurement, signal):
    taps = Tap.get_signal_taps(measurement, signal)
    no_drift_integral_taps = Tap.taps_no_drift_integral(taps)
    return no_drift_integral_taps


def get_taps_diff(measurement, signals):
    result = get_taps_for_function(measurement, signals, get_signal_diff)
    return result


def get_signal_diff(measurement, signal):
    taps = Tap.get_signal_taps(measurement, signal)
    no_drift_integral_taps = Tap.taps_diff(taps)
    return no_drift_integral_taps


def get_taps_for_function(measurement, signals, function):
    taps = []
    for signal in signals:
        taps_i = function(measurement, signal)
        if len(taps_i) > 0:
            taps.append(taps_i)
        else:
            taps = []
            break
    result = taps_reshape(taps)
    return result


def taps_reshape(data):
    """
    input list<broj signala, list<broj tapova, ndarray<broj odbiraka, >>>
    output list<broj tapova, ndarray<broj signal, broj odbiraka, >>
    """
    result = []
    signal_num = len(data)
    tap_num = len(data[0]) if len(data) > 0 else 0
    for tap_index in range(tap_num):
        temp = []
        for signal_index in range(signal_num):
            temp.append(data[signal_index][tap_index])
        result_i = np.asarray(temp)
        result.append(result_i)
    return result


# =============================================================================
# TAPS FUNCTIONS  - vraca listu ndarray
# =============================================================================


def get_taps_convolution_avg(taps):
    if len(taps) == 0: return []
    max_len = Tap.tap_max_len(taps)
    conv = Tap.avg_val_tap(taps, max_len)
    convolutions = taps_convolution(taps, conv)
    result = Tap.signal_to_taps(convolutions, taps)
    return result


def get_taps_convolution_first(taps):
    if len(taps) == 0: return []
    conv = Tap.val_tap(taps, 0)
    convolutions = taps_convolution(taps, conv)
    result = Tap.signal_to_taps(convolutions, taps)
    return result


def get_taps_convolution_last(taps):
    if len(taps) == 0: return []
    conv = Tap.val_tap(taps, -1)
    convolutions = taps_convolution(taps, conv)
    result = Tap.signal_to_taps(convolutions, taps)
    return result


def taps_convolution(taps, conv):
    signals = Tap.concatenate_taps(taps)
    if len(conv.shape) > 1:
        result = signal.convolve2d(signals, conv, 'same')
    else:
        result = np.convolve(signals, conv, 'same')
    return result


def get_taps_auto_convolution(taps):
    if len(taps) == 0: return []
    conv = taps_auto_convolution(taps)
    result = Tap.signal_to_taps(conv, taps)
    return result


def taps_auto_convolution(taps):
    signals = Tap.concatenate_taps(taps)
    if len(signals.shape) > 1:
        result = signal.convolve2d(signals, signals, 'same')
    else:
        result = np.convolve(signals, signals, 'same')
    return result


def get_taps_convolution_single_avg(taps):
    if len(taps) == 0: return []
    max_len = Tap.tap_max_len(taps)
    conv = Tap.avg_val_tap(taps, max_len)
    result = taps_single_convolution(taps, conv)
    return result


def get_taps_convolution_single_first(taps):
    if len(taps) == 0: return []
    conv = Tap.val_tap(taps, -1)
    result = taps_single_convolution(taps, conv)
    return result


def get_taps_convolution_single_last(taps):
    if len(taps) == 0: return []
    conv = Tap.val_tap(taps, -1)
    result = taps_single_convolution(taps, conv)
    return result


def taps_single_convolution(taps, conv):
    results = []
    for tap in taps:
        if len(conv.shape) > 1:
            result = signal.convolve2d(tap, conv, 'same')
        else:
            result = np.convolve(tap, conv, 'same')
        results.append(result)
    return results


def get_taps_single_auto_convolution(taps):
    if len(taps) == 0: return []
    result = taps_single_auto_convolution(taps)
    return result


def taps_single_auto_convolution(taps):
    results = []
    for tap in taps:
        if len(tap.shape) > 1:
            result = signal.convolve2d(tap, tap, 'same')
        else:
            result = np.convolve(tap, tap, 'same')
        results.append(result)
    return results


def get_taps_rfft(taps):
    rfft_taps = Tap.taps_rfft(taps)
    return rfft_taps
