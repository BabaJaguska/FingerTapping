# =============================================================================
# CONVERSIONS
# =============================================================================

class SignalType:
    values = 'signal_values'  # vraca 6, po 3 (x, y, z) i sa palca i sa kaziprsta
    values_idx = 'signal_values_index_only' # vraca 3, x,y,z samo sa kaziprsta
    spherical = 'signal_spherical'  # vraca 6, po 3 (r, fi, te) i sa palca i sa kaziprsta
    amplitude = 'signal_amplitude'  # vraca 2, intenzitet i sa palca i sa kaziprsta
    signed_amplitude = 'signal_signed_amplitude'  # vraca 2, oznaceni intenzitet i sa palca i sa kaziprsta
    square_amplitude = 'signal_square_amplitude'  # vraca 2, kvadrat intenzitet i sa palca i sa kaziprsta
    integral_amplitude = 'signal_integral_amplitude'  # vraca 2, integral intenziteta amplitude i sa palca i sa kaziprsta
    no_drift_integral_amplitude = 'signal_no_drift_integral_amplitude'  # vraca 2, integral intenzitet i sa palca i sa kaziprsta, linearno odstrano odstupanje
    spectrogram = 'signal_spectrogram'  # vraca 256, po 128 spektrogram sa palca i sa kaziprsta
    spectrogram_max = 'signal_max_spectrogram'  # vraca 2, max spektrograma i sa palca i sa kaziprsta
    values_scaled = 'signal_values_scaled'  # vraca 6, po 3 (x, y, z) i sa palca i sa kaziprsta, svaki signal podeqen sa max sih signala tog prsta
    spherical_scaled = 'signal_spherical_scaled'  # vraca 6, po 3 (r, fi, te) i sa palca i sa kaziprsta, r podeljeni sa max r tog prsta
    amplitude_scaled = 'signal_amplitude_scaled'  # vraca 2, intenzitet i sa palca i sa kaziprsta, svaki signal podeqen sa max sih signala tog prsta


class TapType:
    none = 'taps_none'  # ne izdvaja tapove vec vraca isti signal
    taps = 'taps_taps'  # secka na tapove
    normalised_len = 'taps_normalised_len'  # svaki tap se skalira na stretch_len vrednosti
    normalised_max_len = 'taps_normalised_max_len'  # svi tapovi su duzine najduzeg u tom merenju, dopunjeni nulama
    max_len_normalised = 'taps_max_len_normalised'  # svi tapovi su dopunjeni nulama do najduzeg u tom merenju pa onda skalirani na stretch_len vrednosti
    set_len = 'taps_set_len'  # stvi tapovi su duzine max_tap_len, neki su odseceni, a neki dopunji nulama
    double_stretch = 'taps_double_stretch'  # svaki tap skaliran na stretch_len i podeqeno sa maksimalnom apsolutnom vrednoscu tog tapa
    no_drift_integral = 'taps_no_drift_integral'  # za svaki tap je izracunat integral svakog signala, linearno odsecen odsupanje
    diff = 'taps_diff'  # za svaki tap je izracunata razlika susednih vrednosti signala, izvod, svakog signala


class FunctionType:
    none = 'function_none'  # vraca ulazne tapove
    avg_tap = 'convolution_avg_tap'  # skupu sve tapove u jedan odradi konvoluciju sa prosecnim tapom pa ih ponovo isecka na tapove
    first_tap = 'convolution_first_tap'  # skupu sve tapove u jedan odradi konvoluciju sa prvim tapom pa ih ponovo isecka na tapove
    last_tap = 'convolution_last_tap'  # skupu sve tapove u jedan odradi konvoluciju sa poslednjim tapom pa ih ponovo isecka na tapove
    auto = 'convolution_auto'  # skupu sve tapove u jedan odradi konvoluciju sa istim tim signalom pa ih ponovo isecka na tapove
    single_avg_tap = 'convolution_single_avg_tap'  # za svaki tap pojedinacno odradi konvoluciju sa prosecnim tapom
    single_first_tap = 'convolution_single_first_tap'  # za svaki tap pojedinacno odradi konvoluciju sa prvim tapom
    single_last_tap = 'convolution_single_last_tap'  # za svaki tap pojedinacno odradi konvoluciju sa poslednjim tapom
    single_auto = 'convolution_single_auto'  # za svaki tap pojedinacno odradi konvoluciju sa samim sobom
    rfft = 'rfft'  # za svaki tap vraca realni deo FFT


class ConcatenationType:
    none = 'concatenate'  # samo nadoveze tapove
    first = 'concatenate_first'  # nadovezuje tapove tako sto uzima prvi od prvor, pa prvi od drugog, dokle ima tapova, pa onda drugi od prvog, drugi od drugog, ....
    last = 'concatenate_last'  # nadovezuje tapove tako sto na kraj stavi poslednji od prvor, pa poslednji od drugog, dokle ima tapova, pa onda pretposledwi od prvog, pretposlednji od drugog, ....
    min = 'concatenate_min'  # sortira signal po intenzitetu prvo najmanje vrednost prvog tapa, pa najmanja vrednost drugog tapa, ...
    max = 'concatenate_max'  # sortira signal po intenzitetu prvo najveca vrednost prvog tapa, pa najveca vrednost drugog tapa, ...
    first_matrix = 'concatenate_first_matrix'  # isto sto i concatenate_first samo sto svaki tap odseca na max_tap_len, i uzima max_taps tapova
    d3 = 'concatenate_3D'

class AugmentationType: # erm...
    none = 'augmentation_none'
    sliding_taps = 'sliding_taps'

conversion_combinations = [
    [SignalType.values_idx, TapType.none, FunctionType.none, ConcatenationType.none]
]

augmentationType = 'sliding_taps'  # TODO: popravi ovo, kako se koristi klasa? gde ide conversion_combinations
conversion_type = 'create_simple'  # koristi conversion_combinations za obradu

# conversion_type = 'create_full_list' # za obradu pravi listu gde se uzima svaki SignalType, TapType, FunctionType, ConcatenationType, oni koji ne treba da se koriste zakomentarisati
# conversion_type = 'add_all_to_list' # u listu conversion_combinations uz date redove dodaje po jedan koji se uzima  svaki SignalType, TapType, FunctionType, ConcatenationType, oni koji ne treba da se koriste zakomentarisati

# conversion_type = 'create_random_list' # za obradu pravi number_of_conversions kombinacija gde se slucajno uzima SignalType, TapType, FunctionType, ConcatenationType, oni koji ne treba da se koriste zakomentarisati
# conversion_type = 'add_random_list' # u listu conversion_combinations uz date redove dodaje po jedan koji se slucajno uzima svaki SignalType, TapType, FunctionType, ConcatenationType, oni koji ne treba da se koriste zakomentarisati, pravi number_of_conversions kombinacija
number_of_conversions = 1 # koliko kombinacija podataka da se generise. koristi ga 'create_random_list' i 'add_random_list'

# =============================================================================
# CONFIGURATIONS
# =============================================================================

# configuration_type = 'random_attr' # formira number_of_configurations+1 konfiguracija ML algoritama zadatih attribute_range_values skupom. varira sve vrednosti izmedju pretposlednje i poslednje kolone. Uvek vraca i podrazumevane vrednosti, drugu kolonu
configuration_type = 'random_one_attr'  # formira number_of_configurations+1 konfiguracija ML algoritama zadatih attribute_range_values skupom. u jednoj konfiguraciji varira jednu vrednosti izmedju pretposlednje i poslednje kolone. Uvek vraca i podrazumevane vrednosti, drugu kolonu
number_of_configurations = 1
attribute_range_values = (('nConvLayers', 3, 2, 6),
                          ('kernelSize', 7, 8, 60),
                          ('stride', 1, 1, 1),
                          ('constraint', 3, 3, 3),
                          ('nInitialFilters', 16, 32, 128),
                          ('batchSize', 32, 20, 80),
                          ('nDenseUnits', 32, 64, 80),
                          ('dropout_rate1', 0.45, 0.4, 0.6),
                          ('dropout_rate2', 0.45, 0.3, 0.8))

# configuration_type = 'one_attr' # formira vise konfiguracije ML aloritama variranjem jednog, one_attr_name, atributa u intervalu od one_attr_start do one_attr_end sa korakom one_attr_step, ostali atributi dobijaju vrednosti is attribute_default_values
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

number_of_tries_per_configurations = 1  # za svaku konfiguraciju koliko puta da ponovi izracunavanje zbog random vrednsti na pocetku
epochs = 200  # koliko epoha za izracunavanje

# =============================================================================
# TESTS
# =============================================================================

test_type = 'create_simple_tests'  # jedan pacijent se moze naci u samo u jednoj grupi (test, traim, validation)
# test_type = 'create_mixed_tests' # jedan pacijent se moze naci u vise grupa (test, traim, validation)
# test_type = 'create_folded_tests' # 1 pacijent u 1 grupi ali K fold (5 x [train/test])

number_of_tests = 1  # koliko razlicitih odabiranja pacijenata da se koristi

# Split into train, test, val sets
train_percent = 0.9  # procenat za treniranje
test_percent = 0.05  # procenat za testiranje
# validation_percent = 0.1  # procenat za validaciju, ne koristi se

start_time = 0  # pocetak signala koji se posmatra
end_time = 6.715  # kraj signala koji se posmatra, nezavisno od toga koliko signal traje, ako je kraci onda dopuna nulama
def_signal_val = 0  # vrednost za dopunu

samples = round((end_time - start_time) * 200)  # koliko ima odabiraka u signalu

max_taps = 5  # koliko tapova se posmatra, samo za first_matrix
max_tap_len = 100  # kolika je maksimalna duzina tapa. samo za set_len i first_matrix
# samples = max_tap_len * max_taps # koliko ima odabiraka u signalu
tap_stride = 1

stretch_len = 400  # na koliko odbiraka da isteglji signal

# ============================================================================
# GAN
#=============================================================================
# generator
z_dim = 256
n_classes = 4
batch_size = 256
n_epochs = 600
lr = 0.002
device = 'cuda'
display_step = 50
# =============================================================================
# MISCELLANEOUS
# =============================================================================

default_results_path = './results/'  # folder sa rezultatima
default_results_file = 'result.txt'  # fajla sa tekstualnim rezultatima
default_results_csv = 'results.csv'  # fajl sa csv rezultatima
splits_file = './allSplits.txt'  # fajl sa odredjenim granicama tapova
result_in_csv = True

# default_root_path = './data/raw data/' # folder sa podacima koji sadrze i spektrogram
# data_packing_type = 'Zaki' # da li se ucitava spektrogram i nema informacije o koricenoj ruci
default_root_path = './data/raw data1/'  # foler sa podacima koji ne sadrze spektrogram
data_packing_type = 'Minja'  # da li ne ucitava spektrograma i sadrzi informaciju o koriscenoj ruci

show_all = 0  # da li da iscrtava sve dijagrame
decimal_places = 4  # sa koliko decimalnih mesta da prikazuje rezultat

# =============================================================================
# OLD - NOT IN USE
# =============================================================================

root = 'd:/Users/zaki/kod/python/FingerTapping/raw data/'
path = 'd:/Users/zaki/kod/python/FingerTapping/'
