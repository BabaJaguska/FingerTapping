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
    normalised_len = 'taps_normalised_len' # STRETCHUJE NA stretch_len;  sve ovo vazi samo za matrix concatenate
    normalised_max_len = 'taps_normalised_max_len' # padovanje na duzinu max tapa iz TOG signala
    max_len_normalised = 'taps_max_len_normalised' # ako je vece, crop na max len ili stretch_len; ako je manje stretchuj na to
    fixed_len_pad = 'taps_padded_to_fixed_len' # resized or padded to match stretch_len ??
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
    [SignalType.values, TapType.fixed_len_pad, FunctionType.none, ConcatenationType.d3]
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
attribute_range_values = (('nConvLayers', 2,3,4),
                          ('kernelSize', 3,5,7),
                          ('stride', 1, 1, 1),
                          ('constraint', 3, 3, 3),
                          ('nInitialFilters', 8, 16, 32),
                          ('batchSize', 16, 32, 64),
                          ('nDenseUnits', 32, 64, 128),
                          ('dropout_rate1', 0.4, 0.5, 0.6),
                          ('dropout_rate2', 0.4, 0.5, 0.8))

# configuration_type = 'one_attr'
attribute_default_values = (('nConvLayers', 3),
                            ('kernelSize', 7),
                            ('stride', 1),
                            ('constraint', 3),
                            ('nInitialFilters', 16),
                            ('batchSize', 32),
                            ('nDenseUnits', 32),
                            ('dropout_rate1', 0.5),
                            ('dropout_rate2', 0.6))
one_attr_name = 'batchSize'
one_attr_start = 30
one_attr_end = 60
one_attr_step = 3

# =============================================================================
# MODEL TOPOLOGIES
# =============================================================================


# model_topology_type = 'CNNSequentialMLModel'
# model_topology_type = 'CNNSequential2DMLModel'
# model_topology_type = 'CNNRandomMLModel'
# model_topology_type = 'LSTMMLModel'
# model_topology_type = 'MultiHeadedMLModel'
model_topology_type = 'CNNLSTMMLModel'


# =============================================================================
# EVALUATION
# =============================================================================

number_of_tries_per_configurations = 3 # 3 vrtimo isti 
epochs = 200

# =============================================================================
# TESTS
# =============================================================================

test_type = 'create_simple_tests'
# test_type = 'create_mixed_tests'

number_of_tests = 1 # 5 <--- razlicite podele podataka NEMA SMISLA AKO SI STAVILA SEED ali mozes foldove

# Split into train, test, val sets
train_percent = 0.7
test_percent = 0.2
validation_percent = 0.1

start_time = 0
end_time = 10 # KOLIKO MAX DA BUDE SIGNAL, NEVEZANO ZA TO KAD STAJU TAPOVI
def_signal_val = 0

samples = (end_time - start_time) * 200

max_taps = 30
max_tap_len = 600 # nesto drugo; na kolko da secka; ima zero padding ; VIDI STRETCH_LEN
# samples = max_tap_len * max_taps
stretch_len = 600 # ovoliki da ti budu tapovi

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


# =============================================================================
# OLD - NOT IN USE
# =============================================================================

root = 'd:/Users/zaki/kod/python/FingerTapping/raw data/'
path = 'd:/Users/zaki/kod/python/FingerTapping/'
