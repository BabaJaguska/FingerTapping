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
    # all = 'signal_all'


class TapType:
    none = 'taps_none'
    taps = 'taps_taps'
    normalised_len = 'taps_normalised_len'
    normalised_max_len = 'taps_normalised_max_len'
    max_len_normalised = 'taps_max_len_normalised'
    double_stretch = 'taps_double_stretch'
    no_drift_integral = 'taps_no_drift_integral'


class FunctionType:
    none = 'function_none'
    avg_tap = 'convolution_avg_tap'
    first_tap = 'convolution_first_tap'
    last_tap = 'convolution_last_tap'
    auto = 'convolution_auto'
    rfft = 'rfft'


class ConcatenationType:
    none = 'concatenate'
    first = 'concatenate_first'
    last = 'concatenate_last'
    min = 'concatenate_min'
    max = 'concatenate_max'


conversion_combinations = [
    [SignalType.signed_amplitude, TapType.taps, FunctionType.none, ConcatenationType.first],
    [SignalType.values, TapType.taps, FunctionType.none, ConcatenationType.first]
]
conversion_type = 'create_simple'

# conversion_type = 'create_full_list'
# conversion_type = 'add_all_to_list'

# conversion_type = 'create_random_list'
# conversion_type = 'add_random_list'
number_of_conversions = 200

# =============================================================================
# CONFIGURATIONS
# =============================================================================

# configuration_type = 'random_attr'
# configuration_type = 'random_one_attr'
number_of_configurations = 50
attribute_range_values = (('nConvLayers', 3, 3, 6),
                          ('kernelSize', 11, 8, 20),
                          ('stride', 1, 1, 10),
                          ('constraint', 3, 3, 3),
                          ('nInitialFilters', 32, 32, 128),
                          ('batchSize', 16, 16, 50),
                          ('nDenseUnits', 64, 64, 64),
                          ('dropout_rate1', 0.5, 0.2, 0.6),
                          ('dropout_rate2', 0.7, 0.3, 0.75))

configuration_type = 'one_attr'
attribute_default_values = (('nConvLayers', 3),
                            ('kernelSize', 11),
                            ('stride', 1),
                            ('constraint', 3),
                            ('nInitialFilters', 32),
                            ('batchSize', 16),
                            ('nDenseUnits', 64),
                            ('dropout_rate1', 0.5),
                            ('dropout_rate2', 0.7))
one_attr_name = 'kernelSize'
one_attr_start = 8
one_attr_end = 15
one_attr_step = 1

# =============================================================================
# MODEL TOPOLOGIES
# =============================================================================

model_topology_type = 'CNNMLModel'
# model_topology_type = 'CNNSequentialMLModel'
# model_topology_type = 'CNNSequential2DMLModel'
# model_topology_type = 'CNNRandomMLModel'
# model_topology_type = 'LSTMMLModel'
# model_topology_type = 'MultiHeadedMLModel'
# model_topology_type = 'CNNLSTMMLModel'


# =============================================================================
# EVALUATION
# =============================================================================

number_of_tries_per_configurations = 5
epochs = 200

# =============================================================================
# TESTS
# =============================================================================

# test_type = 'create_simple_tests'
test_type = 'create_mixed_tests'

number_of_tests = 2

# Split into train, test, val sets
train_percent = 0.7
test_percent = 0.2
validation_percent = 0.1

start_time = 0
end_time = 10
def_signal_val = 0

# =============================================================================
# MISCELLANEOUS
# =============================================================================

default_results_path = './results/'
default_results_file = 'result.txt'
splits_file = './allSplits.txt'

default_root_path = './data/raw data/'
data_packing_type = 'Zaki'
# default_root_path = './data/raw data1/'
# data_packing_type = 'Minja'

show_all = 0
decimal_places = 4
stretch_len = 100

# =============================================================================
# OLD - NOT IN USE
# =============================================================================

root = 'd:/Users/zaki/kod/python/FingerTapping/raw data/'
path = 'd:/Users/zaki/kod/python/FingerTapping/'
