import numpy as np

import Diagnosis
import Extractor
import Parameters
import Tap


def convert_measurements(measurements, conversions, start, end):
    # converts measurements into signal+diagnosis for the given interval

    signals = []
    diagnoses = []
    for measurement in measurements:
        _signals, _diagnoses = convert_measurement(measurement, start, end, conversions)
        
        if isinstance(_signals, list):
            if not _signals:
                continue
        else:
            if _signals.size == 0:
                continue

            # FIXME: NE VALJA, NE RADI ZA SVE VIDOVE ULAZA
            # if isinstance(_signals, list):
            #     signals += _signals

            # else:
            #     signals += list(_signals)
                
            signals.append(_signals)
                
        
            diagnoses.append( _diagnoses)


    signals, diagnoses = reshape(signals, diagnoses)

    return signals, diagnoses


def convert_measurement(measurement, start, end, conversions, def_val=Parameters.def_signal_val):
    result_taps = []
    concatenation_type = None
    for conversion in conversions:
        signal_type = conversion[0][0] #FIXME: ponekad je conversion lista u listi !??
        #konkretno za 1 conversion signal values, taps none... 
        tap_type = conversion[1] if len(conversion) > 1 else None
        function_type = conversion[2] if len(conversion) > 2 else None
        concatenation_type = conversion[3] if len(conversion) > 3 else None

        signals = get_signals(measurement, signal_type)

        cropped_signals = adjust_signals(measurement, signals, start, end, def_val)

        taps = get_taps(measurement, cropped_signals, tap_type)
   
        function_taps = get_taps_function(taps, function_type)
        concatenated_taps = get_concatenated_taps(function_taps, concatenation_type)
        concatenated_taps = crop_signals(concatenated_taps, 0, Parameters.samples, concatenation_type, def_val)

        if Parameters.ConcatenationType == 'concatenate_3D':
            result_taps += concatenated_taps
        else:
            result_taps.append(concatenated_taps)
        result_signals = concatenate_combinations(result_taps, concatenation_type)
        

    diagnosis = Diagnosis.encode_diagnosis(measurement.diagnosis)
    
    if len(result_signals.shape) > 3: #TODO: DEBUG!! NE BUDE 4D VIDI ZASTO
        diagnosis = np.tile(diagnosis, (result_signals.shape[0],1))

    return result_signals, diagnosis


def get_signals(measurement, signal_type=None):
    result = None

    if signal_type == 'signal_values':
        result = Extractor.get_values(measurement)
    elif signal_type == 'signal_values_index_only':
        result = Extractor.get_values_idx(measurement)
    elif signal_type == 'signal_spherical':
        result = Extractor.get_spherical(measurement)

    elif signal_type == 'signal_amplitude':
        result = Extractor.get_amplitude(measurement)
    elif signal_type == 'signal_signed_amplitude':
        result = Extractor.get_signed_amplitude(measurement)
    elif signal_type == 'signal_diff_signed_amplitude':
        result = Extractor.get_diff_signed_amplitude(measurement)
    elif signal_type == 'signal_square_amplitude':
        result = Extractor.get_square_amplitude(measurement)

    elif signal_type == 'signal_integral_amplitude':
        result = Extractor.get_amplitude_integral(measurement)
    elif signal_type == 'signal_no_drift_integral_amplitude':
        result = Extractor.get_amplitude_no_drift_integral(
            measurement)

    elif signal_type == 'signal_spectrogram':
        result = Extractor.get_spectrogram(measurement)
    elif signal_type == 'signal_max_spectrogram':
        result = Extractor.get_max_spectrogram(measurement)

    elif signal_type == 'signal_values_scaled':
        result = Extractor.get_values_scaled(measurement)
    elif signal_type == 'signal_spherical_scaled':
        result = Extractor.get_spherical_scaled(measurement)
    elif signal_type == 'signal_amplitude_scaled':
        result = Extractor.get_amplitude_scaled(measurement)
    elif signal_type == 'signal_all':
        result = Extractor.get_all(measurement)
    else:
        result = Extractor.get_one_signal(measurement)
    return result


def get_taps(measurement, signals, tap_type=None):
    result = None
    if tap_type == 'taps_none' or tap_type is None:
        result = Extractor.get_tap(measurement, signals)
    elif tap_type == 'taps_taps':
        result = Extractor.get_taps(measurement, signals)
    elif tap_type == 'taps_normalised_len':
        result = Extractor.get_taps_normalised_len(measurement, signals)
    elif tap_type == 'taps_normalised_max_len':
        result = Extractor.get_taps_normalised_max_len(measurement, signals)
    elif tap_type == 'taps_max_len_normalised':
        result = Extractor.get_taps_max_len_normalised(measurement, signals)
    elif tap_type == 'taps_set_len':
        result = Extractor.get_taps_set_len(measurement, signals)
    elif tap_type == 'taps_double_stretch':
        result = Extractor.get_taps_double_stretch(measurement, signals)
    elif tap_type == 'taps_no_drift_integral':
        result = Extractor.get_taps_no_drift_integral(measurement, signals)
    elif tap_type == 'taps_diff':
        result = Extractor.get_taps_diff(measurement, signals)

    return result


def get_taps_function(taps, function_type=None):
    result = None
    if function_type == 'function_none' or function_type is None:
        result = taps
    elif function_type == 'convolution_avg_tap':
        result = Extractor.get_taps_convolution_avg(taps)
    elif function_type == 'convolution_first_tap':
        result = Extractor.get_taps_convolution_first(taps)
    elif function_type == 'convolution_last_tap':
        result = Extractor.get_taps_convolution_last(taps)
    elif function_type == 'convolution_auto':
        result = Extractor.get_taps_auto_convolution(taps)
    elif function_type == 'convolution_single_avg_tap':
        result = Extractor.get_taps_convolution_single_avg(taps)
    elif function_type == 'convolution_single_first_tap':
        result = Extractor.get_taps_convolution_single_first(taps)
    elif function_type == 'convolution_single_last_tap':
        result = Extractor.get_taps_convolution_single_last(taps)
    elif function_type == 'convolution_single_auto':
        result = Extractor.get_taps_single_auto_convolution(taps)
    elif function_type == 'rfft':
        result = Extractor.get_taps_rfft(taps)

    return result


def get_concatenated_taps(taps, concatenation_type=None):
    result = None
    if concatenation_type == 'concatenate' or concatenation_type is None:
        result = Tap.concatenate_taps(taps)
    elif concatenation_type == 'concatenate_first':
        result = Tap.taps_to_ordered_signal(taps)
    elif concatenation_type == 'concatenate_last':
        result = Tap.taps_to_reverse_ordered_signal(taps)
    elif concatenation_type == 'concatenate_min':
        result = Tap.taps_to_min_sorted_signal(taps)
    elif concatenation_type == 'concatenate_max':
        result = Tap.taps_to_max_sorted_signal(taps)
    elif concatenation_type == 'concatenate_first_matrix':
        result = Tap.taps_to_first_matrix_signal(taps)
    elif concatenation_type == 'concatenate_3D':
        result = Tap.concatenate_taps_3D(taps)

    return result


def adjust_signals(measurement, signals, start, end, def_val=0):
    start_index = int(start * measurement.sampling_rate)
    end_index = int(end * measurement.sampling_rate)

    crops = crop_signals(signals, start_index, end_index, def_val)

    return crops


def crop_signals(signals, start_index, end_index, concatenation_type, def_val=0):
    result = []
    if concatenation_type != 'concatenate_3D': 
        for signal in signals:
            crops = signal[..., start_index:end_index] #<--- ODNOSI SE NA ZELJENU DUZINU SIGNALA, NE NA KRAJ TAPKANJA
    
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
        
    else:
        crops = signals
    return crops


def reshape(x, y):
    sizes_x = [len(x)]
    for i in range(len(x[0].shape)):
        sizes_x.append(x[0].shape[i])
    sizes_x = tuple(sizes_x)
    
    if len(sizes_x) > 2:
        x = np.reshape(x, sizes_x)
        x = np.swapaxes(x, -2, -1) #FIXME: ne treba za svaki vid ulaza
    else:
        x = np.array(x)
    # if len(x.shape) > 3:
    #     if x.shape[1] == 1:
    #         x = np.squeeze(x, axis = 1) # pazi ovo da li remeti non-3D pakovanje?
    sizes_y = (len(x), y[0].shape[0])
    
    
    y = np.array(y)
    print('Shape of X: ', x.shape)
    return x, y


def concatenate_combinations(result_taps, concatenation_type):
    """
    input: list<broj kombinacija signala, ndarray<broj signala, broj odbiraka, >>
	output: ndarray<broj signala, broj odbiraka, >
    """
    
    if concatenation_type == 'concatenate_3D': # TODO urediti ovo da struktura bude jednoobrazna
        result = np.array(result_taps)
    else:
    
        signals = []
        for tap in result_taps:
            if len(tap) > 0:
                for signal in tap:
                    signals.append(signal)
            else:
                signals = []
                break
        result = np.asarray(signals) if len(signals) > 0 else []
    return result


# --------------------------------------------------------------------------------------------

def crop_taps(measurement, taps, start, end, def_val):  # TODO ne koristi se
    result = []
    start_index = int(start * measurement.sampling_rate)
    end_index = int(end * measurement.sampling_rate)
    start = 0
    for i in range(len(taps)):
        tap = taps[i]
        end = start + len(tap)
        if i == len(taps) - 1:
            tap = crop_signals(tap, 0, end_index - end, def_val)
        if start_index <= start and end < end_index:
            result.append(tap)
        elif start_index <= start and end >= end_index:
            tmp = crop_signals(tap, 0, end_index - end, def_val)
            result.append(tmp)
        else:
            tmp = crop_signals(tap, start_index - start, end, def_val)
            result.append(tmp)
        start = end
    return result


        