# =============================================================================
# CONVERSIONS
# =============================================================================

class SignalType:
    values = 'signal_values'
    spherical = 'signal_spherical'
    amplitude = 'signal_amplitude'
    signed_amplitude = 'signal_signed_amplitude'
    diff_signed_amplitude = 'signal_diff_signed_amplitude'
    square_amplitude = 'signal_square_amplitude'
    integral_amplitude = 'signal_integral_amplitude'
    no_drift_integral_amplitude = 'signal_no_drift_integral_amplitude'
    spectrogram = 'signal_spectrogram'
    spectrogram_max = 'signal_max_spectrogram'
    values_scaled = 'signal_values_scaled'
    spherical_scaled = 'signal_spherical_scaled'
    amplitude_scaled = 'signal_amplitude_scaled'
#    # all = 'signal_all'


class TapType:
    none = 'taps_none'
    taps = 'taps_taps'
    normalised_len = 'taps_normalised_len'
    normalised_max_len = 'taps_normalised_max_len' # svi u merenju tapovi su duzine najduzeg u tom merenju?
    max_len_normalised = 'taps_max_len_normalised' # svi tapovi duzine 100???
    double_stretch = 'taps_double_stretch'
    no_drift_integral = 'taps_no_drift_integral'


class FunctionType:
    none = 'function_none'
    avg_tap = 'convolution_avg_tap'
    first_tap = 'convolution_first_tap'
    last_tap = 'convolution_last_tap'
    auto = 'convolution_auto'
    single_avg_tap = 'convolution_single_avg_tap'
    single_first_tap = 'convolution_single_first_tap'
    single_last_tap = 'convolution_single_last_tap'
    single_auto = 'convolution_single_auto'
    rfft = 'rfft'


class ConcatenationType:
    none = 'concatenate'
    first = 'concatenate_first'
    last = 'concatenate_last'
    min = 'concatenate_min'
    max = 'concatenate_max'
    first_matrix = 'concatenate_first_matrix'
    d3 = 'concatenate_3D'


conversion_combinations = [
    [SignalType.values, TapType.normalised_len, FunctionType.none, ConcatenationType.d3]
]

conversion_type = 'create_simple'

# conversion_type = 'create_full_list'
# conversion_type = 'add_all_to_list'

# conversion_type = 'create_random_list'
# conversion_type = 'add_random_list'
number_of_conversions = 1

# =============================================================================
# CONFIGURATIONS
# =============================================================================

# configuration_type = 'random_attr'
configuration_type = 'random_one_attr'
number_of_configurations = 1
attribute_range_values = (('nConvLayers', 4, 3, 8),
                          ('kernelSize', 27, 8, 32),
                          ('stride', 1, 1, 10),
                          ('constraint', 3, 3, 3),
                          ('nInitialFilters', 113, 70, 150),
                          ('batchSize', 35, 20, 90),
                          ('nDenseUnits', 64, 64, 64),
                          ('dropout_rate1', 0.5, 0.5, 0.5),
                          ('dropout_rate2', 0.64, 0.5, 0.8))

# configuration_type = 'one_attr'
attribute_default_values = (('nConvLayers', 4),
                            ('kernelSize', 10),
                            ('stride', 1),
                            ('constraint', 3),
                            ('nInitialFilters', 90),
                            ('batchSize', 35),
                            ('nDenseUnits', 64),
                            ('dropout_rate1', 0.5),
                            ('dropout_rate2', 0.64))
one_attr_name = 'batchSize'
one_attr_start = 30
one_attr_end = 60
one_attr_step = 3

# =============================================================================
# MODEL TOPOLOGIES
# =============================================================================

model_topology_type = 'CNNLSTMModel'
# model_topology_type = 'CNNSequentialMLModel'
# model_topology_type = 'CNNSequential2DMLModel'
# model_topology_type = 'CNNRandomMLModel'
# model_topology_type = 'LSTMMLModel'
# model_topology_type = 'MultiHeadedMLModel'
# model_topology_type = 'CNNLSTMMLModel'


# =============================================================================
# EVALUATION
# =============================================================================

number_of_tries_per_configurations = 1
epochs = 200

# =============================================================================
# TESTS
# =============================================================================

test_type = 'create_simple_tests'
# test_type = 'create_mixed_tests'

number_of_tests = 3

# Split into train, test, val sets
train_percent = 0.7
test_percent = 0.2
validation_percent = 0.1

start_time = 0
end_time = 10 # KOLIKO MAX DA BUDE SIGNAL, NEVEZANO ZA TO KAD STAJU TAPOVI
def_signal_val = 0

samples = (end_time - start_time) * 200

max_taps = 30
max_tap_len = 600
# samples = max_tap_len * max_taps

# =============================================================================
# MISCELLANEOUS
# =============================================================================

default_results_path = './results/'

import os
if not os.path.isdir('./results'):
    os.mkdir('./results')    

default_results_file = 'result.txt'
default_results_csv = 'results.csv'
splits_file = './allSplits.txt'

# default_root_path = './data/raw data/'
# data_packing_type = 'Zaki'
default_root_path = './data/raw data1/'
data_packing_type = 'Minja'

show_all = 0
decimal_places = 4
stretch_len = 100

# =============================================================================
# OLD - NOT IN USE
# =============================================================================

root = 'd:/Users/zaki/kod/python/FingerTapping/raw data/'
path = 'd:/Users/zaki/kod/python/FingerTapping/'
