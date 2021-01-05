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
    values_scaled = 'signal_values_scaled'
    spherical_scaled = 'signal_spherical_scaled'
    amplitude_scaled = 'signal_amplitude_scaled'
    all = 'signal_all'


class TapType:
    none = 'taps_none'
    taps = 'taps_taps'
    normalised_len = 'taps_normalised_len'
    normalised_max_len = 'taps_normalised_max_len'
    max_len_normalised = 'taps_max_len_normalised'
    double_stretch = 'taps_double_stretch'
    no_drift_integral = 'taps_no_drift_integral'


combinations = [
    [SignalType.amplitude, TapType.none],
    [SignalType.amplitude, TapType.taps],
    [SignalType.amplitude, TapType.normalised_len],
    [SignalType.amplitude, TapType.normalised_max_len],
    [SignalType.amplitude, TapType.max_len_normalised],
    [SignalType.amplitude, TapType.double_stretch],
    [SignalType.amplitude, TapType.no_drift_integral]
]

# =============================================================================
# CONFIGURATIONS
# =============================================================================

# configuration_type = 'random_attr'
number_of_configurations = 10
attribute_range_values = (('nConvLayers', 4, 2, 5),
                          ('kernelSize', 21, 10, 50),
                          ('stride', 1, 1, 10),
                          ('constraint', 3, 3, 3),
                          ('nInitialFilters', 51, 16, 64),
                          ('batchSize', 27, 16, 64),
                          ('nDenseUnits', 32, 32, 70),
                          ('dropout_rate1', 0.53, 0.5, 0.7),
                          ('dropout_rate2', 0.7, 0.7, 0.7))

configuration_type = 'one_attr'
attribute_default_values = (('nConvLayers', 3),
                            ('kernelSize', 11),
                            ('stride', 1),
                            ('constraint', 3),
                            ('nInitialFilters', 32),
                            ('batchSize', 16),
                            ('nDenseUnits', 64),
                            ('dropout_rate1', 0.6),
                            ('dropout_rate2', 0.7))
one_attr_name = 'dropout_rate1'
one_attr_start = 0.6
one_attr_end = 0.7
one_attr_step = 0.1

# =============================================================================
# MODEL TOPOLOGIES
# =============================================================================

model_topology_type = 'CNNMLModel'
# model_topology_type = 'CNNSequentialMLModel'
# model_topology_type = 'CNNRandomMLModel'
# model_topology_type = 'LSTMMLModel'
# model_topology_type = 'MultiHeadedMLModel'

# =============================================================================
# EVALUATION
# =============================================================================

number_of_tries_per_configurations = 5
epochs = 100

# =============================================================================
# TESTS
# =============================================================================

test_type = 'create_simple_tests'
# test_type = 'create_mixed_tests'

number_of_tests = 5

# Split into train, test, val sets
train_percent = 0.7
test_percent = 0.2
validation_percent = 0.1

start_time = 0
end_time = 10
def_signal_val = 0

load_all = False

# =============================================================================
# MISCELLANEOUS
# =============================================================================

default_results_path = './results/'
#default_root_path = './raw data/'
default_root_path = 'C:/data/icef/tapping/raw data/'
default_results_file = 'result.txt'
splits_file = './allSplits.txt'
data_packing_type = 'Minja' # or 'Zaki'

show_all = 0
decimal_places = 4
stretch_len = 100

# =============================================================================
# OLD - NOT IN USE
# =============================================================================

root = 'd:/Users/zaki/kod/python/FingerTapping/raw data/'
path = 'd:/Users/zaki/kod/python/FingerTapping/'
