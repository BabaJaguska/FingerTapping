import json

import PIL
import numpy as np
from PIL import Image

import Parameters
import Util


def load_all_taps(taps_file=Parameters.splits_file):
    taps = []
    with open(taps_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = json.loads(line)
            temp['allSplitPoints'] = [int(point) for point in temp['allSplitPoints']]
            taps.append(temp)
    return taps


def get_signal_taps(measurement, signal):
    taps = []
    tap_times = measurement.time_tap
    for i in range(len(tap_times) - 1):
        start = tap_times[i]
        end = tap_times[i + 1]
        if (end > start) and (signal.shape[-1] > end):
            tap_values = signal[..., start:end]
            taps.append(tap_values)

    return taps


def stretch_time_taps(taps, new_len=Parameters.stretch_len):
    result = []
    for tap in taps:
        if len(tap.shape) > 1:
            tmp = []
            for signal in tap:
                try:
                    stretched_signal = stretch(signal, new_len)
                    if stretched_signal is not None: tmp.append(stretched_signal)
                except:
                    tmp = []
                    break
            tmp_tap = np.asarray(tmp) if len(tmp) > 0 else []
            result.append(tmp_tap)
        else:
            stretched_signal = stretch(tap, new_len)
            result.append(stretched_signal)
    return result


def stretch(signal, new_len):
    if len(signal.shape) > 1:
        return None  # TODO prosiriti tako da i visedimenzioni mogu da se teglje
    im = Image.fromarray(signal)
    size = (1, new_len)
    array = np.array(im.resize(size, PIL.Image.BICUBIC))
    array_flat = array.flatten()
    return array_flat


def crop_signal_time_taps(taps, new_len=Parameters.stretch_len, default_value=0):
    result = []
    for tap in taps:
        if len(tap) >= new_len:
            new_tap = np.resize(tap, (new_len,))
        else:
            new_tap = np.lib.pad(tap, (0, new_len - len(tap)), 'constant', constant_values=default_value)
        result.append(new_tap)
    return result


def avg_val_tap(taps, tap_len=Parameters.stretch_len):  # TODO izmeniti
    if len(taps) == 0: return []

    diff = np.zeros((taps[0].shape[0], tap_len,)) if len(taps[0].shape) > 1 else np.zeros((tap_len,))
    for tap in taps:
        size = min(tap_len, tap.shape[-1])
        for i in range(size):
            if len(taps[0].shape) > 1:
                a = diff[:, i]
                b = tap[:, i]
                diff[:, i] = a + b
            else:
                a = diff[i]
                b = tap[i]
                diff[i] = a + b
    diff = diff / len(taps)

    return diff


def val_tap(taps, index):
    result = taps[index]
    return result


def diff_taps(taps, diff):
    result = []
    for tap in taps:
        new_tap = tap - diff
        result.append(new_tap)
    return result


def stretch_val_taps(taps, max_val):
    result = []
    for tap in taps:
        new_tap = tap / max_val
        result.append(new_tap)
    return result


def stretch_val_each_taps(taps):
    result = []
    for tap in taps:
        if len(tap.shape) > 1:
            tmp = []
            for signal in tap:
                max_val = max(abs(signal.max()), abs(signal.min()))
                new_tap = signal / max_val if max_val != 0 else signal
                tmp.append(new_tap)
            tmp_tap = np.asarray(tmp) if len(tmp) > 0 else []
            result.append(tmp_tap)
        else:
            max_val = max(abs(tap.max()), abs(tap.min()))
            new_tap = tap / max_val if max_val != 0 else tap
            result.append(new_tap)

    return result


def tap_max_len(taps):
    max_len = 0
    for tap in taps:
        if max_len < tap.shape[-1]:
            max_len = tap.shape[-1]
    return max_len


def tap_max_abs_val(taps):
    max_val = 0
    for tap in taps:
        current = max(abs(tap.max()), abs(tap.min()))
        if max_val < current:
            max_val = current
    return max_val


def taps_integral(taps):
    result = []
    for tap in taps:
        if len(tap.shape) > 1:
            tmp = []
            for signal in tap:
                new_tap = Util.calc_integral(signal)
                tmp.append(new_tap)
            tmp_tap = np.asarray(tmp) if len(tmp) > 0 else []
            result.append(tmp_tap)
        else:
            new_tap = Util.calc_integral(tap)
            result.append(new_tap)
    return result


def taps_no_drift_integral(taps):
    result = []
    for tap in taps:
        if len(tap.shape) > 1:
            tmp = []
            for signal in tap:
                new_tap = Util.calc_no_drift_integral(signal)
                tmp.append(new_tap)
            tmp_tap = np.asarray(tmp) if len(tmp) > 0 else []
            result.append(tmp_tap)
        else:
            new_tap = Util.calc_no_drift_integral(tap)
            result.append(new_tap)

    return result


def taps_rfft(taps):
    result = []
    for tap in taps:
        if len(tap.shape) > 1:
            tmp = []
            for signal in tap:
                new_tap = np.fft.rfft(signal)
                new_tap = abs(new_tap)
                tmp.append(new_tap)
            tmp_tap = np.asarray(tmp) if len(tmp) > 0 else []
            result.append(tmp_tap)
        else:
            new_tap = np.fft.rfft(tap)
            new_tap = abs(new_tap)
            result.append(new_tap)
    return result


def taps_to_ordered_signal(taps, signal=None):
    if signal is not None:
        taps = signal_to_taps(signal, taps)

    max_tap_len = tap_max_len(taps)

    result = []
    for index in range(max_tap_len):
        for k in range(len(taps)):
            tap = taps[k]
            if tap.shape[-1] > index:
                a = tap[:, index] if len(tap.shape) > 1 else tap[index]
                result.append(a)

    result = concatenate_taps(result, False)
    return result


def taps_to_reverse_ordered_signal(taps, signal=None):
    if signal is not None:
        taps = signal_to_taps(signal, taps)

    max_tap_len = tap_max_len(taps)

    result = []
    for index in range(max_tap_len, -1, -1):
        for k in range(len(taps)):
            tap = taps[k]
            if tap.shape[-1] > index:
                a = tap[:, index] if len(tap.shape) > 1 else tap[index]
                result.append(a)

    result.reverse()
    result = concatenate_taps(result, False)
    return result


def taps_to_min_sorted_signal(taps):
    result = taps_to_sorted_signal(taps, False)
    return result


def taps_to_max_sorted_signal(taps):
    result = taps_to_sorted_signal(taps, True)
    return result


def taps_to_sorted_signal(taps, reverse=False):
    result = []
    scalar = False
    for tap in taps:
        if len(tap.shape) > 1:
            tmp = []
            for signal in tap:
                signal1 = np.sort(signal) if not reverse else np.sort(signal)[::-1]
                tmp.append(signal1)
            tmp_tap = np.asarray(tmp) if len(tmp) > 0 else []
            result.append(tmp_tap)
        else:
            new_tap = np.sort(tap) if not reverse else np.sort(tap)[::-1]
            result.append(new_tap)
            scalar = True
    result = concatenate_taps(result, scalar)
    return result


def concatenate_taps(taps, scalar=True):
    if len(taps) > 0:
        tl = len(taps[0].shape)
        if tl > 1:  # TODO proveriti da li ovo lepo radi sa visedimenzionim nizovima
            result = np.concatenate(taps, axis=-1)
        elif tl == 1:
            if scalar:
                result = np.concatenate(taps)
            else:
                result = np.asarray(taps)
                result = np.swapaxes(result, 0, 1)
        else:
            result = np.asarray(taps)
    else:
        result = []
    return result


def signal_to_taps(signal, taps):  # TODO da li radi za visedimenzione?
    result = []
    start = 0
    for tap in taps:
        end = start + tap.shape[-1]
        sublist = signal[:, start:end] if len(signal.shape) > 1 else signal[start:end]
        if len(sublist) > 0:
            result.append(sublist)
        start = end
    return result


def crop_val_taps(taps, min_val, max_val):  # TODO da li radi?
    result = []
    for tap in taps:
        new_tap = np.zeros((len(tap),))
        for i in range(len(tap)):
            val = tap[i]
            if val > max_val:
                new_tap[i] = max_val
            elif val < min_val:
                new_tap[i] = min_val
            else:
                new_tap[i] = tap[i]
        result.append(new_tap)
    return result
