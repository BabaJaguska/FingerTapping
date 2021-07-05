import traceback
from math import comb
from random import randint

import PIL
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn import svm
from tqdm import tqdm

import Parameters
import Signal
from ml.ArtefactExtractor import calc_max_frequency, calc_median_frequency
from ml.ArtefactFilter import get_filtered_selected_names, get_filters
from ml.ArtefactSelectorGenerator import random_artefacts

s_size = 200


def main():
    signals = Signal.load_all(Parameters.default_root_path)

    for signal in tqdm(signals):
        observed_signal = signal.gyro2Vec

        if len(signal.time_tap) > 0:
            try:
                val, plot_val, stretch_val, max_frequency, max_val, median_frequency, median_pow = transform(
                    observed_signal)

                plt.figure(figsize=(16, 5))
                plt.plot(stretch_val)
                plt.legend(['specter'])

                plt.axvline(x=max_frequency, color='b')
                plt.axvline(x=median_frequency, color='g')
                plt.plot([s_size], [5], 'r')

                plt.title(signal.diagnosis + ' ' + signal.file[20:42])
                # plt.show()
                plt.savefig('./results/' + signal.diagnosis + '_' + signal.file[20:42] + 'S.png')
                plt.close()

            except:
                print("An exception occurred {}".format(signal.diagnosis + ' ' + signal.file[20:42]))
                traceback.print_exc()
    return


def transform(observed_signal):
    val = observed_signal
    val = abs(np.fft.rfft(val))
    val = val[1:-1]

    max_frequency, max_val = calc_max_frequency(val)

    median_frequency, median_pow = calc_median_frequency(val)

    plot_val = val / len(val)

    stretch_val = stretch(plot_val, s_size)
    stretch_max_val = max(stretch_val)

    stretch_val = stretch_val * max_val / stretch_max_val
    max_stretch_frequency = max_frequency * s_size
    median_stretch_frequency = median_frequency * s_size

    return val, plot_val, stretch_val, max_stretch_frequency, max_val, median_stretch_frequency, median_pow


def test_sin():
    Fs = 8000
    f = 100
    omega = 2 * np.pi * f / Fs
    a = 16

    sample1 = 14000
    x1 = np.arange(sample1)
    y1 = a * np.sin(omega * x1)

    sample2 = 29000
    x2 = np.arange(sample2)
    y2 = a * np.sin(omega * x2)

    t1 = transform(y1)
    t1 = t1[2]
    max_t1 = max(t1)

    t2 = transform(y2)
    t2 = t2[2]
    max_t2 = max(t2)

    plt.plot(t1, 'r')
    plt.plot(t2, 'g')

    for i in range(100):
        sample_i = 8000 + i * 160
        xi = np.arange(sample_i)
        yi = a * np.sin(omega * xi)
        val, plot_val, stretch_val, max_frequency, max_val, median_frequency, median_pow = transform(yi)
        max_val_i = max(val)
        max_plot_val_i = max(plot_val)
        max_stretch_val_i = max(stretch_val)
        print(
            '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(sample_i, max_val_i, max_plot_val_i, max_stretch_val_i,
                                                    max_frequency, max_val, median_frequency, median_pow))

    plt.xlabel('sample(n)')
    plt.ylabel('voltage(V)')
    plt.show()

    return


def additiveNoise1(y1, yg):
    t = randint(0, len(y1) - len(yg) - 1)
    result = []
    cnt = 0
    for i in range(len(y1)):
        if i < t or i >= t + len(yg):
            result.append(y1[i])
        else:
            result.append(y1[i] + yg[cnt])
            cnt = cnt + 1
    return result


def additiveNoise2(y1, yg):
    t = randint(0, len(y1) - len(yg) - 1)
    result = []
    cnt = 0
    for i in range(len(y1)):
        if i < t:
            result.append(y1[i])
        elif cnt < len(yg):
            result.append(y1[t] + yg[cnt])
            cnt = cnt + 1
        else:
            result.append(y1[i - cnt])
    return result


def test_sin1():
    Fs = 200
    f = 4
    omega = 2 * np.pi * f / Fs
    a = 16

    sample1 = 16 * 200
    print_range = 200
    x1 = np.arange(sample1)
    y1 = a * np.sin(omega * x1)
    val1 = abs(np.fft.rfft(y1))
    val1 = val1[1:print_range]

    gomega = 2 * np.pi * f / Fs
    b = a / 4
    glitch_time = int(2 * np.pi / gomega) + 1

    xg = np.arange(glitch_time)
    yg = b * np.sin(gomega * xg)

    y2 = additiveNoise1(y1, yg)
    val2 = abs(np.fft.rfft(y2))
    val2 = val2[1:print_range]

    y3 = additiveNoise2(y1, yg)
    val3 = abs(np.fft.rfft(y3))
    val3 = val3[1:print_range]

    plt.figure(1)
    plt.plot(val1, 'r')
    plt.plot(val2, 'g')
    plt.plot(val3, 'b')

    plt.xlabel('Hz')
    plt.ylabel('specter')

    plt.figure(2)
    # plt.plot(y1, 'r')
    # plt.plot(y2, 'g')
    plt.plot(y3, 'b')
    # plt.plot(yg, 'y')

    plt.xlabel('time')
    plt.ylabel('amplitude')
    plt.show()

    return


def stretch(signal, new_len):
    if len(signal.shape) > 1:
        return None  # TODO prosiriti tako da i visedimenzioni mogu da se teglje
    im = Image.fromarray(signal)
    size = (1, new_len)
    array = np.array(im.resize(size, PIL.Image.BICUBIC))
    array_flat = array.flatten()
    return array_flat


def test_svm():
    # Our dataset and targets
    X = np.c_[(.4, -.7),
              (-1.5, -1),
              (-1.4, -.9),
              (-1.3, -1.2),
              (-1.1, -.2),
              (-1.2, -.4),
              (-.5, 1.2),
              (-1.5, 2.1),
              (1, 1),
              # --
              (1.3, .8),
              (1.2, .5),
              (.2, -2),
              (.5, -2.4),
              (.2, -2.3),
              (0, -2.7),
              (1.3, 2.1)].T
    Y = [0] * 8 + [1] * 8

    # figure number
    fignum = 1

    # fit the model
    for kernel in ('linear', 'poly', 'rbf'):
        clf = svm.SVC(kernel=kernel, gamma=2)
        clf.fit(X, Y)

        # plot the line, the points, and the nearest vectors to the plane
        plt.figure(fignum, figsize=(4, 3))
        plt.clf()

        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                    facecolors='none', zorder=10, edgecolors='k')
        plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
                    edgecolors='k')

        plt.axis('tight')
        x_min = -3
        x_max = 3
        y_min = -3
        y_max = 3

        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(XX.shape)
        plt.figure(fignum, figsize=(4, 3))
        plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired, shading='auto')
        plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                    levels=[-.5, 0, .5])

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.xticks(())
        plt.yticks(())
        fignum = fignum + 1
    plt.show()
    return


def test_random():
    results = random_artefacts(34, 40000)
    for artefact in results:
        print('{0:b}'.format(artefact.selector))


def plot_1_2():
    signals = Signal.load_all(Parameters.default_root_path)

    for signal in tqdm(signals):

        if len(signal.time_tap) > 0:
            try:
                signal_x = signal.gyro1x  # [signal.time_tap[0]:signal.time_tap[-1]]
                signal_y = signal.gyro1y  # [signal.time_tap[0]:signal.time_tap[-1]]
                signal_z = signal.gyro1z  # [signal.time_tap[0]:signal.time_tap[-1]]

                x = np.arange(0, len(signal_x))
                plt.figure(figsize=(16, 5))
                plt.plot(x, signal_x, 'r', x, signal_y, 'g', x, signal_z, 'b')
                plt.legend(['1x', '1y', '1z'])

                plt.title(signal.diagnosis + ' ' + signal.initials + ' ' + signal.file[20:42])
                # plt.show()
                plt.savefig(
                    './results/' + signal.diagnosis + '_' + signal.initials + '_' + signal.file[20:42] + '1.png')
                plt.close()

            except:
                print("An exception occurred {}".format(signal.diagnosis + ' ' + signal.file[20:42]))
                traceback.print_exc()
    return


def print_names():
    selectors = [
        0b100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000000010000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000

    ]
    for selector in selectors:
        names = get_filtered_selected_names(selector)
        print(names)

    return


def print_selectors():
    selectors = [
        ['gyro2y_acc_rms', 'gyro2x_power_avg', 'gyro1x_max_val', 'gyro2Vec_median_frequency',
         'gyro2y_max_speed_avg', 'gyro1z_max_acc_std']

    ]
    for selector in selectors:
        ints = get_filters(selector)
        print(bin(ints))
    return


def calc():
    M = 100
    N = 4
    sum1 = 0
    for i in range(N + 1):
        sum1 = sum1 + comb(M, i)
    print(sum1)


print_names()
calc()
print_selectors()
#test_sin1()
