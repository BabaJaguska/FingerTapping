import traceback

from tqdm import tqdm

import Extractor
import Parameters
import Signal
import Signal2Test
import Tap
import Util


def main():
    measurements = Signal.load_all(Parameters.default_root_path)
    start = 0
    end = 20
    def_val = 0

    for measurement in tqdm(measurements):
        result = []

        result1 = [measurement.gyro2VecSign]
        _result1 = Extractor.list_to_array(result1)
        result.append(result1)
        result2 = Extractor.get_values(measurement)
        _result2 = Extractor.list_to_array(result2)
        result.append(result2)
        result3 = Extractor.get_spherical(measurement)
        _result3 = Extractor.list_to_array(result3)
        result.append(result3)
        result4 = Extractor.get_amplitude(measurement)
        _result4 = Extractor.list_to_array(result4)
        result.append(result4)
        result5 = Extractor.get_signed_amplitude(measurement)
        _result5 = Extractor.list_to_array(result5)
        result.append(result5)
        result6 = Extractor.get_diff_signed_amplitude(measurement)
        _result6 = Extractor.list_to_array(result6)
        result.append(result6)
        result7 = Extractor.get_square_amplitude(measurement)
        _result7 = Extractor.list_to_array(result7)
        result.append(result7)
        result8 = Extractor.get_amplitude_integral(measurement)
        _result8 = Extractor.list_to_array(result8)
        result.append(result8)
        result9 = Extractor.get_amplitude_no_drift_integral(measurement)
        _result9 = Extractor.list_to_array(result9)
        result.append(result9)
        result10 = Extractor.get_spectrogram(measurement)
        _result10 = Extractor.list_to_array(result10)
        result.append(result10)
        result11 = Extractor.get_max_spectrogram(measurement)
        _result11 = Extractor.list_to_array(result11)
        result.append(result11)
        result12 = Extractor.get_values_scaled(measurement)
        _result12 = Extractor.list_to_array(result12)
        result.append(result12)
        result13 = Extractor.get_spherical_scaled(measurement)
        _result13 = Extractor.list_to_array(result13)
        result.append(result13)
        result14 = Extractor.get_amplitude_scaled(measurement)
        _result14 = Extractor.list_to_array(result14)
        result.append(result14)
        result15 = Extractor.get_all(measurement)
        _result15 = Extractor.list_to_array(result15)
        result.append(result15)

        results = []
        for r in result:
            Util.concatenate_lists(results, r)
        _result16 = Extractor.list_to_array(results)

        cropped_signals1 = Signal2Test.adjust_signals(measurement, _result1, start, end, def_val)
        cropped_signals2 = Signal2Test.adjust_signals(measurement, _result2, start, end, def_val)
        cropped_signals16 = Signal2Test.adjust_signals(measurement, _result16, start, end, def_val)

        taps = Tap.get_signal_taps(measurement, result)

        if len(taps) > 0:
            try:
                print()
            except:
                print("An exception occurred {}".format(measurement.diagnosis + ' ' + measurement.file[20:42]))
                traceback.print_exc()

    return


main()
