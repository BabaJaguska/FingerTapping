from math import sqrt

import numpy as np

from Artefact import Artefact
from Util import calc_no_drift_integral_poly, calc_diff


def extract(measurements):
    results = []
    for measurement in measurements:
        print(measurement.id)
        artefacts = extract_artefacts(measurement)
        if artefacts is not None: results.append(artefacts)
    return results


def extract_artefacts(measurement, use_taps=True):
    if use_taps and ((measurement.time_tap is None) or len(measurement.time_tap) == 0): return None

    dict_values = dict()

    time_tap = measurement.time_tap

    for function in functions:
        for get in getters:
            signals, name = get(measurement)
            vals = function(signals, use_taps, time_tap)
            for val in vals:
                val.update_name(name)
                dict_values[val.name] = val

    name = measurement.file
    description = measurement.initials
    decision = measurement.diagnosis
    result = Artefact(name, description, None, dict_values, decision)
    return result


def speed_info(signals, use_taps, time_tap):
    result = generic_signal_info(signals, use_taps, time_tap, trans, 'speed')
    return result


def speed_info2(signals, use_taps, time_tap):
    result = generic_taps_signal_info(signals, use_taps, time_tap, trans, 'speed')
    return result


def speed_info3(signals, use_taps, time_tap):
    result = generic_wobbling_signal_info(signals, use_taps, time_tap, trans, 'speed')
    return result


def acc_info(signals, use_taps, time_tap):
    result = generic_signal_info(signals, use_taps, time_tap, diff, 'acc')
    return result


def acc_info2(signals, use_taps, time_tap):
    result = generic_taps_signal_info(signals, use_taps, time_tap, diff, 'acc')
    return result


def acc_info3(signals, use_taps, time_tap):
    result = generic_wobbling_signal_info(signals, use_taps, time_tap, diff, 'acc')
    return result


def angle_info(signals, use_taps, time_tap):
    result = generic_signal_info(signals, use_taps, time_tap, integral, 'angle')
    return result


def angle_info2(signals, use_taps, time_tap):
    result = generic_taps_signal_info(signals, use_taps, time_tap, integral, 'angle')
    return result


def angle_info3(signals, use_taps, time_tap):
    result = generic_wobbling_signal_info(signals, use_taps, time_tap, integral, 'angle')
    return result


def power_info(signals, use_taps, time_tap):
    result = generic_signal_info(signals, use_taps, time_tap, pow2, 'power')
    return result


def power_info2(signals, use_taps, time_tap):
    result = generic_taps_signal_info(signals, use_taps, time_tap, pow2, 'power')
    return result


def power_info3(signals, use_taps, time_tap):
    result = generic_wobbling_signal_info(signals, use_taps, time_tap, pow2, 'power')
    return result


def generic_signal_info(signals, use_taps, time_tap, function, prefix):
    if use_taps:
         # val = signals[time_tap[0]:time_tap[-1]]
         val = signals
    else:
        val = signals

    val = function(val, time_tap)

    min_val, max_val, avg_val, rms_val, crest_val, std_val, parp_val = standard_info(val)

    result = (
        ExtractionInfo(prefix + '_min', min_val), 
        ExtractionInfo(prefix + '_max', max_val),
        ExtractionInfo(prefix + '_avg', avg_val), 
        ExtractionInfo(prefix + '_rms', rms_val),
        ExtractionInfo(prefix + '_crest', crest_val),
        ExtractionInfo(prefix + '_std', std_val),
        ExtractionInfo(prefix + '_parp', parp_val))
    return result


def generic_taps_signal_info(signals, use_taps, time_tap, function, prefix):
    if use_taps:
        taps = get_signal_taps(signals, time_tap)
    else:
        taps = [signals]

    val = []
    for tap in taps:
        tap = function(tap, time_tap)
        max_i = max(tap)
        val.append(max_i)

    min_val, max_val, avg_val, rms_val, crest_val, std_val, parp_val = standard_info(val)

    result = (
        ExtractionInfo('max_' + prefix + '_min', min_val),
        ExtractionInfo('max_' + prefix + '_max', max_val),
        ExtractionInfo('max_' + prefix + '_avg', avg_val), 
        ExtractionInfo('max_' + prefix + '_rms', rms_val),
        ExtractionInfo('max_' + prefix + '_crest', crest_val),
        ExtractionInfo('max_' + prefix + '_std', std_val),
        ExtractionInfo('max_' + prefix + '_parp', parp_val))
    return result


def generic_wobbling_signal_info(signals, use_taps, time_tap, function, prefix):
    if use_taps:
        taps = get_signal_taps(signals, time_tap)
    else:
        taps = [signals]

    taps2 = []
    for tap in taps:
        tap = function(tap, time_tap)
        taps2.append(tap)

    angles, areas = calc_max_wobbling(taps2)
    
    angle_min_val, angle_max_val, angle_agv_val, angle_rms_val, \
    angle_crest_val, angle_std_val, angle_papr_val = standard_info(angles)
    
    area_min_val, area_max_val, area_agv_val, area_rms_val,\
    area_crest_val, area_std_val, area_papr_val = standard_info(areas)

    result = (
        ExtractionInfo(prefix + '_wangle_avg', angle_agv_val), ExtractionInfo(prefix + '_wangle_rms', angle_rms_val),
        ExtractionInfo(prefix + '_wangle_crest', angle_crest_val),
        ExtractionInfo(prefix + '_wangle_std', angle_std_val),
        ExtractionInfo(prefix + '_warea_avg', area_agv_val), ExtractionInfo(prefix + '_warea_rms', area_rms_val),
        ExtractionInfo(prefix + '_warea_crest', area_crest_val),
        ExtractionInfo(prefix + '_warea_std', area_std_val))
    return result


def taps_info(signals, use_taps, time_tap):
    time_tap = time_tap

    val = []
    prev = -1
    for tap in time_tap:
        if prev < 0:
            prev = tap
            continue
        dif = tap - prev
        val.append(dif)
        prev = tap

    min_val, max_val, avg_val, rms_val, crest_val, std_val, parp_val = standard_info(val)

    result = (
        ExtractionInfo('taps_min', min_val), ExtractionInfo('taps_max', max_val),
        ExtractionInfo('taps_avg', avg_val), ExtractionInfo('taps_rms', rms_val),
        ExtractionInfo('taps_crest', crest_val), ExtractionInfo('taps_std', std_val),
        ExtractionInfo('taps_parp', parp_val))
    return result


def specter_info(signals, use_taps, time_tap):
    if use_taps:
        # val = signals[time_tap[0]:time_tap[-1]]
        val = signals
    else:
        val = signals

    dc_value, delta_frequency, frequency_ratio, frequency_ratio1,\
        max_frequency, max_val, median_frequency, median_frequency1,\
            median_frequency2, median_pow = standard_specter_info(val, time_tap)

    result = (
        ExtractionInfo('max_val', max_val), ExtractionInfo('max_frequency', max_frequency),
        ExtractionInfo('delta_frequency', delta_frequency), ExtractionInfo('median_frequency', median_frequency),
        ExtractionInfo('median_pow', median_pow), ExtractionInfo('frequency_ratio', frequency_ratio),
        ExtractionInfo('median_frequency1', median_frequency1), ExtractionInfo('median_frequency2', median_frequency2),
        ExtractionInfo('dc_value', dc_value), ExtractionInfo('frequency_ratio1', frequency_ratio1)
    )
    return result


def specter_info2(signals, use_taps, time_tap):
    if use_taps:
        taps = get_signal_taps(signals, time_tap)
    else:
        taps = [signals]

    val = []
    for tap in taps:
        dc_value, delta_frequency, frequency_ratio, frequency_ratio1,\
            max_frequency, max_val, median_frequency, median_frequency1, \
                median_frequency2, median_pow = standard_specter_info(tap, time_tap)
        val.append(median_frequency)

    min_val, max_val, avg_val, rms_val, crest_val, std_val, parp_val = standard_info(val)

    result = (
        # ExtractionInfo('median_min', min_val), ExtractionInfo('median_max', max_val),
        # ExtractionInfo('median_avg', avg_val), ExtractionInfo('median_rms', rms_val),
        # ExtractionInfo('median_crest', crest_val),
        # ExtractionInfo('median_std', std_val),ExtractionInfo('median_parp', parp_val)
        ExtractionInfo('median_change', parp_val),
    )
    return result


def standard_specter_info(val, time_tap):
    val = abs(np.fft.rfft(val))
    dc_value = val[0]
    val = val[1:-1]  # izbaci jednosmernu komponentu
    max_frequency, max_val = calc_max_frequency(val)
    median_frequency, median_pow = calc_median_frequency(val)
    delta_frequency = (median_frequency - max_frequency) / median_frequency if median_frequency > 0 else 0
    num_taps = len(time_tap)
    taps_len = len(val)
    cadence = num_taps / taps_len
    frequency_ratio = median_frequency / cadence
    frequency_ratio1 = max_frequency / cadence
    median_frequency1, median_frequency2, median_pow1 = calc_median_frequency2(val)
    return dc_value, delta_frequency, frequency_ratio, frequency_ratio1,\
        max_frequency, max_val, median_frequency, median_frequency1, \
            median_frequency2, median_pow


def calc_median_frequency(data):
    integral = sum(data)
    sum1 = 0
    i = 0
    for i in range(len(data)):
        sum1 = sum1 + data[i]
        if sum1 * 2 >= integral:
            break
    median_frequency = i / len(data)
    median_specter_power = integral / len(data)
    return median_frequency, median_specter_power


def calc_median_frequency2(data):
    integral = sum(data)
    sum1 = 0

    median_frequency_1 = None
    median_frequency_2 = None
    for i in range(len(data)):
        sum1 = sum1 + data[i]
        if sum1 >= integral / 3 and median_frequency_1 is None:
            median_frequency_1 = i
        elif sum1 >= integral * 2 / 3 and median_frequency_2 is None:
            median_frequency_2 = i
            break

    median_frequency_1 = median_frequency_1 / len(data)
    median_frequency_2 = median_frequency_2 / len(data)

    median_specter_power = integral / len(data)
    return median_frequency_1, median_frequency_2, median_specter_power


def calc_max_frequency(data):
    max_frequency = 0
    max_val = -1
    for i in range(len(data)):
        if data[i] > max_val:
            max_val = data[i]
            max_frequency = i
    max_frequency = max_frequency / len(data)
    max_val = max_val / len(data)
    return max_frequency, max_val


def standard_info(data, start_index=None, end_index=None):
    if start_index is None: start_index = 0
    if end_index is None: end_index = -1

    values = data[start_index:end_index]

    if len(values) == 0: return 0, 0, 0, 0, 0

    min_val = min(values)

    max_val = max(values)

    avg_val = sum(values) / len(values)

    std_val = np.std(values)

    ss = 0
    for val in values:
        ss = ss + val * val
    ms = ss / len(values)
    rms_val = sqrt(ms)

    crest_val = max_val / rms_val if rms_val > 0 else 0
    parp_val = crest_val * crest_val

    return min_val, max_val, avg_val, rms_val, crest_val, std_val, parp_val


def get_signal_taps(signal, time_tap):
    taps = []
    tap_times = time_tap
    for i in range(len(tap_times) - 1):
        start = tap_times[i]
        end = tap_times[i + 1]
        if (end > start) and (signal.shape[-1] > end):
            tap_values = signal[..., start:end]
            taps.append(tap_values)

    return taps


def calc_max_wobbling(taps):
    angles = []
    areas = []

    prev = len(taps[0])
    max_val_prev = max(taps[0])
    max_index_prev = np.where(taps[0] == max_val_prev)[0][0]
    for i in range(1, len(taps)):
        tap = taps[i]
        max_val = max(tap)
        max_index = np.where(tap == max_val)[0][0]
        angle = (max_val - max_val_prev) / (max_index + prev - max_index_prev)
        area = ((max_val + max_val_prev) / 2) * (max_index + prev - max_index_prev)
        angles.append(angle)
        areas.append(area)

        prev = len(tap)
        max_val_prev = max_val
        max_index_prev = max_index

    if len(angles) == 0: angles.append(0)
    if len(areas) == 0: areas.append(0)

    return angles, areas


functions1 = [speed_info, acc_info, angle_info2, power_info, taps_info, specter_info]  # TODO namerno stoji angle_info2
functions2 = [speed_info2, acc_info2, angle_info2, power_info2, taps_info, specter_info2]
functions3 = [speed_info3, acc_info3, angle_info3, power_info3, taps_info, specter_info]

functions1_2 = [speed_info, acc_info, angle_info, power_info,
                taps_info, specter_info, speed_info2, acc_info2,
                angle_info2, power_info2, specter_info2]

functions1_2_3 = [speed_info, acc_info, angle_info, power_info, taps_info, 
                  specter_info, speed_info2, acc_info2,
                  angle_info2, power_info2, specter_info2, 
                  speed_info3, acc_info3, angle_info3, power_info3]

functionsTest = [angle_info, angle_info2, angle_info3]

functions = functions1_2


def get1x(measurement):
    return measurement.gyro1x, 'gyro1x'


def get1y(measurement):
    return measurement.gyro1y, 'gyro1y'


def get1z(measurement):
    return measurement.gyro1z, 'gyro1z'


def get1Vec(measurement):
    return measurement.gyro1Vec, 'gyro1Vec'


def get2x(measurement):
    return measurement.gyro2x, 'gyro2x'


def get2y(measurement):
    return measurement.gyro2y, 'gyro2y'


def get2z(measurement):
    return measurement.gyro2z, 'gyro2z'


def get2Vec(measurement):
    return measurement.gyro2Vec, 'gyro2Vec'


getters = [get1x, get1y, get1z, get1Vec, get2x, get2y, get2z, get2Vec]


class ExtractionInfo:
    def __init__(self, name, value):
        self.name = name
        self.value = value

        return

    def update_name(self, extension):
        self.name = extension + '_' + self.name
        return


def trans(val, time_tap=[]):
    return val


def diff(val, time_tap=[]):
    return calc_diff(val)


def integral(val, time_tap):
    return calc_no_drift_integral_poly(val, time_tap)


def pow2(val, time_tap=[]):
    return val * val


def print_all(artefacts, file_name='./results/raw_data.txt'):
    a0 = artefacts[0]
    keys = a0.dict_values.keys()

    with open(file_name, 'a') as file:
        header = 'name\tdescription'
        for key in keys:
            header = header + '\t' + key
        header = header + '\tresult\n'
        file.write(str(header))

        for artefact in artefacts:
            line = artefact.name + '\t' + artefact.description
            for key in keys:
                line = line + '\t' + str(artefact.dict_values[key].value)
            line = line + '\t' + artefact.result + '\n'
            file.write(str(line))

        file.close()

    return
