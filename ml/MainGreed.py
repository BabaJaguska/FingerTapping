import datetime
from random import randint

import Parameters
import Signal
from ml import ArtefactExtractor, ArtefactSelectorGenerator, ArtefactEvaluatorGenerator, ArtefactEvaluator, \
    ArtefactTestGenerator, ArtefactFilter, ArtefactNormalisator
from ml.ArtefactEvaluator import log


def simulation_init():
    extract = False
    if extract:
        measurements = Signal.load_all(Parameters.default_root_path)
        Signal.show_signals_info(measurements, plot=Parameters.show_all)
        artefacts = ArtefactExtractor.extract(measurements)
        print(len(artefacts))
    else:
        artefacts = ArtefactExtractor.load()

    artefacts = ArtefactNormalisator.normalise(artefacts)
    print(len(artefacts))
    test = ArtefactTestGenerator.generate()
    print(test)
    evaluator = ArtefactEvaluatorGenerator.get_evaluator()
    print(evaluator)
    best_names = [
        ['gyro2y_acc_rms', 'gyro2x_power_avg', 'gyro1x_max_val', 'gyro2Vec_median_frequency',
         'gyro2y_max_speed_avg', 'gyro1z_max_acc_std']
    ]
    best_score = 205
    return best_names, best_score, [artefacts, test, evaluator]


def expand(simulation_params, input_parameters, input_score):
    artefacts = simulation_params[0]
    test = simulation_params[1]
    evaluator = simulation_params[2]

    prev_names = []
    prev_score = -1

    candidate_names = input_parameters
    candidate_score = input_score

    cnt = 0  # todo dodato da bi se zavrsilo u razumno vreme
    used_selectors = set()

    while candidate_score > prev_score or (
            candidate_score == prev_score and len(candidate_names) > len(prev_names) and cnt < 5):
        cnt = 0 if candidate_score > prev_score else cnt + 1

        prev_names, prev_score = candidate_names, candidate_score

        ArtefactFilter.used_artefacts = ArtefactFilter.used_artefacts_all_1_2_3_e
        filtered_artefacts = ArtefactFilter.filtering(artefacts)
        log(ArtefactFilter.get_filtered_names())
        initial_selectors = ArtefactFilter.get_filters(prev_names)
        number_of_ones = randint(2, 4)  # todo menja 1-3 artefakta
        selectors = ArtefactSelectorGenerator.select(filtered_artefacts, selection_type='CHANGE_UP_TO',
                                                     number_of_ones=number_of_ones, number_of_results=1000,
                                                     initial_selectors=initial_selectors)

        # todo moze se zakomentarisati ako je potrebno da se ponovo radi nad istim podacima
        selectors = remove_used(selectors, used_selectors)

        print(selectors)
        results = ArtefactEvaluator.evaluate(filtered_artefacts, selectors, test, evaluator)
        ArtefactEvaluator.print_best(results)

        best_selectors = ArtefactEvaluator.get_bests(results)

        candidate_names = []
        for best_selector in best_selectors:
            candidate_names.append(ArtefactFilter.get_selected_names(best_selector.selector.selector))
        candidate_score = best_selectors[0].get_best_val()

    return prev_names, prev_score


def reduce(simulation_params, input_parameters, input_score):
    artefacts = simulation_params[0]
    test = simulation_params[1]
    evaluator = simulation_params[2]

    ArtefactFilter.used_artefacts = get_names_set(input_parameters)

    filtered_artefacts = ArtefactFilter.filtering(artefacts)
    log(ArtefactFilter.get_filtered_names())
    number_of_ones = randint(6, 7)  # todo ostavlja 5-6 artefakta
    selectors = ArtefactSelectorGenerator.select(filtered_artefacts, selection_type='UP_TO_NUM_OF_ONES',
                                                 number_of_ones=number_of_ones, number_of_results=10000)
    print(selectors)
    results = ArtefactEvaluator.evaluate(filtered_artefacts, selectors, test, evaluator)
    ArtefactEvaluator.print_best(results)

    best_selectors = ArtefactEvaluator.get_bests(results)

    candidate_names = []
    for best_selector in best_selectors:
        candidate_names.append(ArtefactFilter.get_selected_names(best_selector.selector.selector))
    candidate_score = best_selectors[0].get_best_val()

    return candidate_names, candidate_score


def get_names_set(list_list_name):
    name_set = set()
    result = []
    for list_name in list_list_name:
        for name in list_name:
            if name in name_set:
                name_set.add(name)
            else:
                result.append(name)
                name_set.add(name)
    return result


def remove_used(selectors, used_selectors):
    result = []
    for selector in selectors:
        if selector.selector not in used_selectors:
            result.append(selector)
            used_selectors.add(selector.selector)
    return result


def main():
    best_parameters, best_score, simulation_params = simulation_init()

    end_time = datetime.datetime.now() + datetime.timedelta(hours=10)

    while datetime.datetime.now() < end_time:
        iteration_start = datetime.datetime.now()

        expansion_parameters, expansion_score = expand(simulation_params, best_parameters, best_score)

        reduction_parameters, reduction_score = reduce(simulation_params, expansion_parameters, expansion_score)

        if best_score < reduction_score:
            best_parameters = reduction_parameters
            best_score = reduction_score

        iteration_end = datetime.datetime.now()
        iteration_duration = iteration_end - iteration_start

        log('{}, {}, {}\n'.format(iteration_duration, best_parameters, best_score), './results/logBS')
    return


main()
