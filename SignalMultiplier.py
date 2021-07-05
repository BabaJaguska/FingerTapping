from random import uniform

import numpy as np

import Parameters
import Signal


def multiply_measurement(measurements):
    results = []
    i = 0
    for measurement in measurements:
        multiplicated_measurements = multiply_and_noise_measurement(measurement, Parameters.multiplication_factor,
                                                                    Parameters.noise)
        for new_measurements in multiplicated_measurements:
            taped_measurements = taps_measurement(new_measurements, Parameters.number_of_taps_in_measurements,
                                                  Parameters.stride_for_taps_in_measurements)
            for t in taped_measurements:
                results.append(t)
        print('{} of {} files multiplied'.format(i, len(measurements)))
        i = i + 1
    return results


def multiply_and_noise_measurement(measurement, multiplication_factor, noise):
    result = [measurement]
    for i in range(multiplication_factor):
        temp = measurement.copy()
        temp = add_noise_to_measurement(temp, noise)
        result.append(temp)

    return result


def add_noise_to_measurement(temp, noise):
    temp.fsr = add_noise(temp.fsr, noise)

    # angular velocity
    # thumb
    temp.gyro1x = add_noise(temp.gyro1x, noise)
    temp.gyro1y = add_noise(temp.gyro1y, noise)
    temp.gyro1z = add_noise(temp.gyro1z, noise)
    temp.gyro1Vec = np.sqrt(np.square(temp.gyro1x) +
                            np.square(temp.gyro1y) +
                            np.square(temp.gyro1z))
    temp.gyro1VecSign = Signal.signed_amplitude(temp.gyro1x, temp.gyro1y, temp.gyro1z)

    # forefinger
    temp.gyro2x = add_noise(temp.gyro2x, noise)
    temp.gyro2y = add_noise(temp.gyro2y, noise)
    temp.gyro2z = add_noise(temp.gyro2z, noise)
    temp.gyro2Vec = np.sqrt(np.square(temp.gyro2x) +
                            np.square(temp.gyro2y) +
                            np.square(temp.gyro2z))
    temp.gyro2VecSign = Signal.signed_amplitude(temp.gyro2x, temp.gyro2y, temp.gyro2z)

    return temp


def add_noise(vector, noise):
    min_val = min(vector)
    max_val = max(vector)
    delta = abs(max_val - min_val) * noise
    for i in range(len(vector)):
        val = vector[i]
        diff = uniform(-delta, delta)
        if isinstance(val, np.uint16):
            if diff < 0:
                diff = 0
            else:
                diff = int(diff)
        val = val + diff
        vector[i] = val
    return vector


def taps_measurement(measurement, number_of_taps_in_signal=None, stride=1):
    if number_of_taps_in_signal is None: return [measurement]
    result = []

    tap_times = measurement.time_tap
    if len(tap_times) == 0: return result

    start = 0
    end = number_of_taps_in_signal if len(tap_times) > number_of_taps_in_signal else len(tap_times) - 1
    while end < len(tap_times):
        temp = measurement.copy()
        temp = extract_taps(temp, start, end)
        result.append(temp)

        start = start + stride
        end = end + stride

    return result


def extract_taps(temp, start, end):
    start_index = temp.time_tap[start]
    end_index = temp.time_tap[end]

    # force
    temp.fsr = temp.fsr[start_index:end_index] if len(temp.fsr) > 0 else []

    # angular velocity
    # thumb
    temp.gyro1x = temp.gyro1x[start_index:end_index] if len(temp.gyro1x) > 0 else []
    temp.gyro1y = temp.gyro1y[start_index:end_index] if len(temp.gyro1y) > 0 else []
    temp.gyro1z = temp.gyro1z[start_index:end_index] if len(temp.gyro1z) > 0 else []
    temp.gyro1Vec = temp.gyro1Vec[start_index:end_index] if len(temp.gyro1Vec) > 0 else []
    temp.gyro1VecSign = temp.gyro1VecSign[start_index:end_index] if len(temp.gyro1VecSign) > 0 else []

    # forefinger
    temp.gyro2x = temp.gyro2x[start_index:end_index] if len(temp.gyro2x) > 0 else []
    temp.gyro2y = temp.gyro2y[start_index:end_index] if len(temp.gyro2y) > 0 else []
    temp.gyro2z = temp.gyro2z[start_index:end_index] if len(temp.gyro2z) > 0 else []
    temp.gyro2Vec = temp.gyro2Vec[start_index:end_index] if len(temp.gyro2Vec) > 0 else []
    temp.gyro2VecSign = temp.gyro2VecSign[start_index:end_index] if len(temp.gyro2VecSign) > 0 else []

    # thumb spectrogram WVD
    temp.spectrogram_t = temp.spectrogram_t[start_index:end_index] if len(temp.spectrogram_t) > 0 else []

    # forefinger spectrogram WVD
    temp.spectrogram_i = temp.spectrogram_i[start_index:end_index] if len(temp.spectrogram_i) > 0 else []

    temp.time = temp.time[start_index:end_index] if len(temp.time) > 0 else []  # time
    temp.time_tap = temp.time_tap[start:(end + 1)]  # list of taps start/end time
    temp.ttap_start = 0  # single value, when the actual signal started SECONDS
    temp.ttap_stop = len(temp.gyro1x) / temp.sampling_rate  # single value, when the actual signal stopped SECONDS
    temp.length = len(temp.gyro1x)

    return temp
