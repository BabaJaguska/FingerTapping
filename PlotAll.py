import traceback

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import Extractor
import Parameters
import Signal
import Signal2Test
import Tap


def main():
    signals = Signal.load_all(Parameters.default_root_path)

    for signal in tqdm(signals):
        observed_signal = signal.gyro2VecSign

        taps = Tap.get_signal_taps(signal, observed_signal)

        if len(taps) > 0:
            try:
                taps_c = np.concatenate(taps)
                taps_c_2 = taps_c * taps_c

                max_len = Tap.tap_max_len(taps)

                crop_taps = Tap.crop_signal_time_taps(taps, max_len)

                stretch_taps = Tap.stretch_time_taps(taps)

                stretch_crop_taps = Tap.stretch_time_taps(crop_taps)

                crop_taps_avg = Tap.avg_val_tap(crop_taps, max_len)
                crop_taps_diff = Tap.diff_taps(crop_taps, crop_taps_avg)

                stretch_taps_avg = Tap.avg_val_tap(stretch_taps)
                stretch_taps_diff = Tap.diff_taps(stretch_taps, stretch_taps_avg)

                double_stretch_crop_taps = Tap.stretch_val_each_taps(stretch_crop_taps)

                double_stretch_taps = Tap.stretch_val_each_taps(stretch_taps)

                taps_integral = Tap.taps_no_drift_integral(taps)

                stretch_taps_integral = Tap.taps_no_drift_integral(stretch_taps)

                double_stretch_taps_integral = Tap.taps_no_drift_integral(double_stretch_taps)

                taps_convolution_avg = Extractor.get_taps_convolution_avg(taps)
                taps_convolution_first = Extractor.get_taps_convolution_first(taps)
                taps_convolution_last = Extractor.get_taps_convolution_last(taps)
                taps_self_convolution = Extractor.get_taps_auto_convolution(taps)

                taps_convolution_single_avg = Extractor.get_taps_convolution_single_avg(taps)
                taps_convolution_single_first = Extractor.get_taps_convolution_single_first(taps)
                taps_convolution_single_last = Extractor.get_taps_convolution_single_last(taps)
                taps_self_single_convolution = Extractor.get_taps_single_auto_convolution(taps)

                double_stretch_taps_convolution_avg = Extractor.get_taps_convolution_avg(double_stretch_taps)
                double_stretch_taps_convolution_first = Extractor.get_taps_convolution_first(double_stretch_taps)
                double_stretch_taps_convolution_last = Extractor.get_taps_convolution_last(double_stretch_taps)
                double_stretch_taps_self_convolution = Extractor.get_taps_auto_convolution(double_stretch_taps)

                rfft = Tap.taps_rfft(stretch_taps)

                crop_signal = np.reshape(Signal2Test.crop_signals([Tap.concatenate_taps(taps)], 0, 1600, 0), 1600)
                crop_taps = Tap.signal_to_taps(crop_signal, taps)

                taps_ordered = [Tap.taps_to_ordered_signal(crop_taps, crop_signal)]
                taps_reverse_ordered = [Tap.taps_to_reverse_ordered_signal(crop_taps, crop_signal)]
                taps_min_ordered = [Tap.taps_to_min_sorted_signal(crop_taps)]
                taps_max_ordered = [Tap.taps_to_max_sorted_signal(crop_taps)]
                taps_matrix = [Tap.taps_to_first_matrix_signal(crop_taps, 100, 30)]

                taps_diff = Tap.taps_diff(taps)
                taps_diff_convolution_avg = Extractor.get_taps_convolution_avg(taps_diff)

                plot_signal = taps_diff_convolution_avg

                plot_taps(plot_signal, signal, plot_all=False)
            except:
                print("An exception occurred {}".format(signal.diagnosis + ' ' + signal.file[20:42]))
                traceback.print_exc()
    return


def plot_signals(plot_signal, signal, interval=1600):
    plot_val = Signal2Test.crop_signals([plot_signal], 0, interval, 0)
    plot_val = plot_val.flatten()

    plt.figure(figsize=(16, 5))
    plt.plot(plot_val)
    plt.legend(['taps'])
    tap_times = signal.time_tap
    for tap_time in tap_times:
        if tap_time < interval:
            plt.axvline(x=tap_time, color='b')
    plt.title(signal.diagnosis + ' ' + signal.file[20:42])
    # plt.show()
    plt.savefig('./results/' + signal.diagnosis + '_' + signal.file[20:42] + 'S.png')
    plt.close()


def plot_taps(taps, signal, plot_all=False, interval=1600):
    plot_val = np.concatenate(taps)

    if plot_all:
        interval = len(plot_val)

    plot_val = Signal2Test.crop_signals([plot_val], 0, interval, 0)

    plot_val = plot_val.flatten()

    plt.figure(figsize=(16, 5))
    plt.plot(plot_val)
    plt.legend(['taps'])

    tap_time = 0
    for tap in taps:
        if tap_time < interval:
            plt.axvline(x=tap_time, color='b')
        tap_time = tap_time + len(tap)

    name = signal.diagnosis + ' ' + signal.initials + ' ' + signal.date + ' ' + signal.time_of_measurement
    plt.title(name)
    # plt.show()
    file_name = signal.diagnosis + '_' + signal.initials + '_' + signal.date + '_' + signal.time_of_measurement
    plt.savefig('./results/' + file_name + '.png')
    plt.close()


def plot_nd(spectrogram, signal, plot_all=False, interval=1600, suffix="S"):
    import matplotlib.pyplot as plt

    h = spectrogram[:, :interval]

    fig = plt.figure(figsize=(16, 5))

    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    plt.imshow(h)
    ax.set_aspect('equal')

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    name = signal.diagnosis + ' ' + signal.initials + ' ' + signal.date + ' ' + signal.time_of_measurement
    plt.title(name)
    # plt.show()
    file_name = signal.diagnosis + '_' + signal.initials + '_' + signal.date + '_' + signal.time_of_measurement
    plt.savefig('./results/' + file_name + suffix + '.png')
    plt.close()
    return


def tap_to_tap(taps):
    signal = Signal2Test.crop_signals(taps, 0, 1600, 0)
    result = Tap.taps_to_ordered_signal(taps, signal)

    return result


main()
