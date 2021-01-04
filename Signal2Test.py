import numpy as np

import Diagnosis
import Extractor
import Parameters


def convert_signals1(signals, start, end, signal_type, tap_type):
    # converts signal into signal+diagnosis for the given interval

    x = []  # signals
    y = []  # diagnoses
    for signal in signals:
        _x, _y = convert_signal(signal, start, end, signal_type, tap_type)
        if len(_x) > 0:
            x.append(_x)
            y.append(_y)

    x, y = reshape(x, y)

    return x, y


def convert_signals(signals, combinations, start, end):
    # converts signal into signal+diagnosis for the given interval

    x = []  # signals
    y = []  # diagnoses
    for signal in signals:
        _x, _y = convert_signal(signal, start, end, combinations)
        if len(_x) > 0:
            x.append(_x)
            y.append(_y)

    x, y = reshape(x, y)

    return x, y


def convert_signal(signal, start, end, combinations, def_val=Parameters.def_signal_val):
    taps = []
    for combination in combinations:
        signal_type = combination[0]
        tap_type = combination[1]
        signals = get_signals(signal, signal_type)
        _taps, _cnt = get_taps(signal, signals, tap_type)
        if _cnt == 0:
            for _tap in _taps:
                taps.append(_tap)
        else:
            taps = []
            break
    result = adjust(signal, taps, start, end, def_val)
    return result


def get_signals(signal, signal_type=None):
    result = None

    if signal_type == 'signal_values': result = Extractor.get_values(signal)
    if signal_type == 'signal_spherical': result = Extractor.get_spherical(signal)

    if signal_type == 'signal_amplitude':  result = Extractor.get_amplitude(signal)
    if signal_type == 'signal_signed_amplitude':  result = Extractor.get_signed_amplitude(signal)
    if signal_type == 'signal_diff_signed_amplitude':  result = Extractor.get_diff_signed_amplitude(signal)
    if signal_type == 'signal_square_amplitude':  result = Extractor.get_square_amplitude(signal)

    if signal_type == 'signal_integral_amplitude':  result = Extractor.get_amplitude_integral(signal)
    if signal_type == 'signal_no_drift_integral_amplitude':  result = Extractor.get_amplitude_no_drift_integral(signal)

    if signal_type == 'signal_spectrogram':  result = Extractor.get_spectrogram(signal)

    if signal_type == 'signal_values_scaled': result = Extractor.get_values_scaled(signal)
    if signal_type == 'signal_spherical_scaled': result = Extractor.get_spherical_scaled(signal)
    if signal_type == 'signal_amplitude_scaled':  result = Extractor.get_amplitude_scaled(signal)
    if signal_type == 'signal_all':  result = Extractor.get_all(signal)

    return result


def get_taps(signal, signals, tap_type=None):
    result = None
    cnt = 0
    if tap_type == 'taps_none': result, cnt = signals, cnt
    if tap_type == 'taps_taps': result, cnt = Extractor.get_taps(signal, signals)
    if tap_type == 'taps_normalised_len': result, cnt = Extractor.get_taps_normalised_len(signal, signals)
    if tap_type == 'taps_normalised_max_len': result, cnt = Extractor.get_taps_normalised_max_len(signal, signals)
    if tap_type == 'taps_max_len_normalised': result, cnt = Extractor.get_taps_max_len_normalised(signal, signals)
    if tap_type == 'taps_double_stretch': result, cnt = Extractor.get_taps_double_stretch(signal, signals)
    if tap_type == 'taps_no_drift_integral': result, cnt = Extractor.get_taps_no_drift_integral(signal, signals)

    return result, cnt


def adjust(signal, data, start, end, def_val=0):
    start_index = int(start * signal.sampling_rate)
    end_index = int(end * signal.sampling_rate)

    crops = crop(data, start_index, end_index, def_val)

    diagnosis = Diagnosis.encode_diagnosis(signal.diagnosis)
    return crops, diagnosis


def crop(data, start_index, end_index, def_val=0):
    result = []
    for d in data:
        crops = d[..., start_index:end_index]

        l1 = crops.shape[len(crops.shape) - 1]
        s = end_index - start_index - l1
        if s > 0:
            padding = []
            for i in range(len(crops.shape) - 1):
                padding.append((0, 0))
            padding.append((0, s))
            padding = tuple(padding)
            crops = np.lib.pad(crops, padding, 'constant', constant_values=def_val)
        if len(crops.shape) > 1:
            result.append(crops)  # TODO nadovezuje ih u 2D
        else:
            result.append([crops])
    crops = np.concatenate(result) if len(result) > 0 else []
    return crops


def reshape(x, y):
    sizes_x = [len(x)]
    for i in range(len(x[0].shape)):
        sizes_x.append(x[0].shape[i])
    sizes_x = tuple(sizes_x)
    x = np.reshape(x, sizes_x)
    x = np.swapaxes(x, 1, -1)
    sizes_y = (len(x), y[0].shape[0])
    y = np.reshape(y, sizes_y)
    print('Shape of X: ', x.shape)
    return x, y
