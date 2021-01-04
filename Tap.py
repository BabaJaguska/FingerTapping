import json

import PIL
import numpy as np
from PIL import Image

import Extractor
import Parameters


def load_all_taps(taps_file=Parameters.splits_file):
    taps = []
    with open(taps_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            temp = json.loads(line)
            temp['allSplitPoints'] = [int(point) for point in temp['allSplitPoints']]
            taps.append(temp)
    return taps


def get_taps(signal, signal_values):
    taps = []
    tap_times = signal.time_tap
    for i in range(len(tap_times) - 1):
        start = tap_times[i]
        end = tap_times[i + 1]
        if (end > start) and (signal_values.shape[-1] > end):
            tap_values = signal_values[..., start:end]
            taps.append(tap_values)

    return taps


def stretch_time_taps(taps, new_len=Parameters.stretch_len):
    result = []
    for tap in taps:
        try:
            if len(tap.shape) > 1:
                break  # TODO prosiriti tako da i visedimenzioni mogu da se teglje
            im = Image.fromarray(tap)
            size = (1, new_len)
            array = np.array(im.resize(size, PIL.Image.BICUBIC))
            array_flat = array.flatten()
            result.append(array_flat)
        except:
            result = []
            break

    return result


def crop_time_taps(taps, new_len=Parameters.stretch_len, default_value=0):
    result = []
    for tap in taps:
        if len(tap) > new_len:
            new_tap = np.resize(tap, (new_len,))
        else:
            new_tap = np.lib.pad(tap, (0, new_len - len(tap)), 'constant', constant_values=default_value)
        result.append(new_tap)
    return result


def avg_val_tap(taps, tap_len=Parameters.stretch_len):
    diff = np.zeros((tap_len,))
    if len(taps) > 0:
        for j in range(len(taps)):
            size = min(tap_len, taps[j].shape[0])
            for i in range(size):
                diff[i] = diff[i] + taps[j][i]
        diff = diff / len(taps)

    return diff


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
        max_val = max(abs(tap.max()), abs(tap.min()))
        new_tap = tap / max_val if max_val != 0 else tap
        result.append(new_tap)
    return result


def crop_val_taps(taps, min_val, max_val):
    result = []
    for tap in taps:
        new_tap = np.zeros((len(tap),))
        for i in range(len(tap)):
            val = tap[i]
            if val > max_val:
                new_tap[i] = max_val
            else:
                if val < min_val:
                    new_tap[i] = min_val
                else:
                    new_tap[i] = tap[i]
        result.append(new_tap)
    return result


def tap_max_len(taps):
    max_len = 0
    for tap in taps:
        if max_len < tap.shape[0]:
            max_len = tap.shape[0]
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
        new_tap = Extractor.calc_integral(tap)
        result.append(new_tap)
    return result


def taps_no_drift_integral(taps):
    result = []
    for tap in taps:
        new_tap = Extractor.calc_no_drift_integral(tap)
        result.append(new_tap)
    return result
