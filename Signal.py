import math
import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from scipy import io
from tqdm import tqdm

import Diagnosis
import Parameters
import Tap


class Signal:
    def __init__(self, file, fsr,
                 gyro1x, gyro1y, gyro1z, gyro2x, gyro2y, gyro2z, spectrogram_t, spectrogram_i,
                 tap_task, time, time_tap, ttap_start, ttap_stop, diagnosis, initials, date, time_of_measurement):
        # file
        self.file = file

        # force
        self.fsr = fsr

        # angular velocity
        # thumb
        self.gyro1x = gyro1x
        self.gyro1y = gyro1y
        self.gyro1z = gyro1z
        self.gyro1Vec = np.sqrt(np.square(self.gyro1x) +
                                np.square(self.gyro1y) +
                                np.square(self.gyro1z))
        self.gyro1VecSign = signed_amplitude(self.gyro1x, self.gyro1y, self.gyro1z)

        # forefinger
        self.gyro2x = gyro2x
        self.gyro2y = gyro2y
        self.gyro2z = gyro2z
        self.gyro2Vec = np.sqrt(np.square(self.gyro2x) +
                                np.square(self.gyro2y) +
                                np.square(self.gyro2z))
        self.gyro2VecSign = signed_amplitude(self.gyro2x, self.gyro2y, self.gyro2z)

        # thumb spectrogram WVD
        self.spectrogram_t = spectrogram_t  # np.swapaxes(spectrogram_t, 0, 1)

        # forefinger spectrogram WVD
        self.spectrogram_i = spectrogram_i  # np.swapaxes(spectrogram_i, 0, 1)

        # other
        self.sampling_rate = 200  # sampling rate [Hz]
        self.tap_task = tap_task  # LHEO/LHEC/RHEO/RHEC (left or right hand/eyes open or closed)
        self.time = time  # time
        self.time_tap = time_tap  # list of taps start/end time
        self.ttap_start = ttap_start  # single value, when the actual signal started SECONDS
        self.ttap_stop = ttap_stop  # single value, when the actual signal stopped SECONDS
        self.diagnosis = diagnosis  # PD, PSP, MSA, CTRL
        self.initials = initials  # person name and surname initials
        self.date = date  # date of recording
        self.time_of_measurement = time_of_measurement  # what time that date
        self.length = len(gyro1x)

    def plot_signal(self, tmin, tmax):
        # gyro1
        plt.figure(figsize=(16, 5))
        plt.plot(self.time, self.gyro1x)
        plt.plot(self.time, self.gyro1y)
        plt.plot(self.time, self.gyro1z)
        plt.plot(self.time, self.gyro1Vec)
        plt.plot(self.time, self.gyro1VecSign)
        plt.legend(['GyroThumbX', 'GyroThumbY', 'GyroThumbZ', 'GyroThumbVec', 'GyroThumbVecSign'])
        plt.axvline(x=tmin, color='b')
        plt.axvline(x=tmax, color='r')
        plt.xlim(tmin, tmax)
        plt.title('Gyro THUMB data: ' + self.diagnosis + ' ' + self.file[20:42])
        plt.show()

        # gyro2
        plt.figure(figsize=(16, 5))
        plt.plot(self.time, self.gyro2x)
        plt.plot(self.time, self.gyro2y)
        plt.plot(self.time, self.gyro2z)
        plt.plot(self.time, self.gyro2Vec)
        plt.plot(self.time, self.gyro2VecSign)
        plt.legend(['GyroIndexX', 'GyroIndexY', 'GyroIndexZ', 'GyroIndexVec', 'GyroIndexVecSign'])
        plt.axvline(x=tmin, color='b')
        plt.axvline(x=tmax, color='r')
        plt.xlim(tmin, tmax)
        plt.title('Gyro INDEX data: ' + self.diagnosis + ' ' + self.file[20:42])
        plt.show()

        # force
        plt.figure(figsize=(16, 5))
        plt.plot(self.time, self.fsr)
        plt.xlim(tmin, tmax)
        plt.axvline(x=tmin, color='b')
        plt.axvline(x=tmax, color='r')
        plt.title('Normalized FSR: ' + self.file)
        plt.show()

    def get_signal_info(self):
        temp = {'lenFsr': len(self.fsr), 'lenGyroThumb': len(self.gyro1x), 'lenGyroForefinger': len(self.gyro2x),
                'lenTime': len(self.time)}
        temp['MATCHING_LENGTHS'] = len(set(temp.values())) == 1
        temp['durationInSecs'] = self.length / self.sampling_rate
        return temp

    def transform_spherical(self):
        x1 = self.gyro1x
        y1 = self.gyro1y
        z1 = self.gyro1z

        x2 = self.gyro2x
        y2 = self.gyro2y
        z2 = self.gyro2z

        def transpher(x, y, z):
            r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            mask = np.logical_and(np.equal(x, 0), np.equal(y, 0))

            x_masked = ma.masked_array(x, mask)
            y_masked = ma.masked_array(y, mask)
            z_masked = ma.masked_array(z, mask)

            phi = np.arccos(x_masked / np.sqrt(np.square(x_masked) + np.square(y_masked)))
            theta = np.arccos(z_masked / np.sqrt(np.square(x_masked) + np.square(y_masked) + np.square(z_masked)))

            phi = ma.filled(phi, 0)
            theta = ma.filled(theta, 0)

            return r, phi, theta

        r1, phi1, theta1 = transpher(x1, y1, z1)
        r2, phi2, theta2 = transpher(x2, y2, z2)

        return r1, phi1, theta1, r2, phi2, theta2

    def __str__(self):
        temp = self.get_signal_info()
        return str(temp)

    def __repr__(self):
        return str(self)


# Distribution of classes?
def plot_class_distribution(sigs):
    diagnoses = [sig.diagnosis for sig in sigs]

    cnts = []
    text = 'There are '
    all_diagnoses = Diagnosis.get_diagnosis_names()
    for diagnosis in all_diagnoses:
        diag = np.sum([d == diagnosis for d in diagnoses])
        cnts.append(diag)
        text = text + '{} {},'.format(diag, diagnosis)

    text = text + ' subjests'
    print(text)

    plt.bar(all_diagnoses, cnts)
    plt.title('Number of signals recorded by diagnosis')
    plt.show()
    return


def show_signals_info(signals, plot=1):
    print('INFO:')
    print('There are a total of {} files'.format(len(signals)))
    temp = [s.get_signal_info()['MATCHING_LENGTHS'] for s in signals]
    if len(set(temp)) == 1:
        print('All signals contain gyro and fsr data of the same length')
    else:
        print('Some files contain data of unequal lengths')

    # plot
    # plot a random signal
    if plot == 1:
        i = np.random.randint(0, len(signals))
        signals[i].plot_signal(0, 10)

    return


def load(root, directory, file):
    recording = io.loadmat(root + directory + '/' + file)

    sig = recording['Recording']

    # fsr = sig['vectorThumb'][0][0][0] #uzeto ubrzanje palca
    fsr = sig['vectorForefinger'][0][0][0]  # uzeto ubrzanje kaziprsta

    gyro1x = sig['gyroThumb'][0][0][0]
    gyro1y = sig['gyroThumb'][0][0][1]
    gyro1z = sig['gyroThumb'][0][0][2]
    gyro2x = sig['gyroForefinger'][0][0][0]
    gyro2y = sig['gyroForefinger'][0][0][1]
    gyro2z = sig['gyroForefinger'][0][0][2]

    ThumbWVD = sig['ThumbWVD'][0][0]
    IndexWVD = sig['IndexWVD'][0][0]

    tap_task = 'RHEO'  # ????
    time = sig['time'][0][0][0]
    time_tap = []
    ttapstart = time[0]
    ttapstop = time[len(time) - 1]
    diagnosis = directory
    if file[2].isdigit():
        initials = file[0:2]
        date = file[23:33]
        time_of_measurement = file[34:42]
    else:
        initials = file[0:3]
        date = file[25:35]
        time_of_measurement = file[36:44]

    temp = Signal(file, fsr, gyro1x, gyro1y, gyro1z, gyro2x, gyro2y, gyro2z, ThumbWVD, IndexWVD, tap_task, time,
                  time_tap, ttapstart, ttapstop, diagnosis, initials, date, time_of_measurement)

    return temp


def load_all(root=Parameters.default_root_path, taps_file=Parameters.splits_file):
    signals = load_all_signals(root)
    all_taps = Tap.load_all_taps(taps_file)
    for signal in signals:
        file_name = signal.file
        file_name = file_name[-31:-12]
        time_tap = [tap['allSplitPoints'] for tap in all_taps if tap['id'][-19:] == file_name]
        signal.time_tap = time_tap if len(time_tap) == 0 else time_tap[0]
    return signals


def load_all_signals(root=Parameters.default_root_path):
    _, dirs, _ = os.walk(root).__next__()
    signals = []
    for current_dir in tqdm(dirs):
        _, subdirs, files = os.walk(root + current_dir).__next__()
        signals = signals + [load(root, current_dir, file) for file in files]
        for subdir in subdirs:
            dir_name = root + current_dir + '/' + subdir + '/'
            _, _, files1 = os.walk(dir_name).__next__()
            signals = signals + [load(root, current_dir, subdir + '/' + file) for file in files1]
    return signals


def signed_amplitude(xvals, yvals, zvals):  # TODO da li je ovo ispravno?
    result = []
    for i in range(len(xvals)):
        x = xvals[i]
        y = yvals[i]
        z = zvals[i]
        pos = 0
        neg = 0
        x2 = x ** 2
        y2 = y ** 2
        z2 = z ** 2
        r = math.sqrt(x2 + y2 + z2)
        if x > 0:
            pos = pos + x2
        else:
            neg = neg + x2
        if y > 0:
            pos = pos + y2
        else:
            neg = neg + y2
        if z > 0:
            pos = pos + z2
        else:
            neg = neg + z2
        if neg > pos:
            r = -r
        result.append(r)
    result = np.array(result)
    return result
