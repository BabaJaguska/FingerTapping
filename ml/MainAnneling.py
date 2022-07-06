import datetime
from math import exp

from numpy.random import rand

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


def two_phase_move(simulation_params, current_names):
    artefacts = simulation_params[0]
    test = simulation_params[1]
    evaluator = simulation_params[2]

    # phase one
    ArtefactFilter.used_artefacts = ArtefactFilter.used_artefacts_all_1_2_3_e
    filtered_artefacts = ArtefactFilter.filtering(artefacts)
    log(ArtefactFilter.get_filtered_names())
    initial_selectors = ArtefactFilter.get_filters(current_names)
    selectors = ArtefactSelectorGenerator.select(filtered_artefacts, selection_type='CHANGE_UP_TO',
                                                 number_of_ones=3, number_of_results=10000,
                                                 initial_selectors=initial_selectors)
    print(selectors)
    results = ArtefactEvaluator.evaluate(filtered_artefacts, selectors, test, evaluator)
    ArtefactEvaluator.print_best(results)
    best_selectors = ArtefactEvaluator.combine_best_n(results, 2)
    # phase two
    ArtefactFilter.used_artefacts = ArtefactFilter.get_selected_names(best_selectors)
    filtered_artefacts = ArtefactFilter.filtering(artefacts)
    log(ArtefactFilter.get_filtered_names())
    selectors = ArtefactSelectorGenerator.select(filtered_artefacts, selection_type='UP_TO_NUM_OF_ONES',
                                                 number_of_ones=7, number_of_results=10000)
    print(selectors)
    results = ArtefactEvaluator.evaluate(filtered_artefacts, selectors, test, evaluator)
    ArtefactEvaluator.print_best(results)
    best_selectors = ArtefactEvaluator.get_best_n(results, 1)
    candidate_names = ArtefactFilter.get_selected_names(best_selectors[0].selector.selector)
    candidate_score = best_selectors[0].get_best_val()
    return [candidate_names], candidate_score


def one_phase_move(simulation_params, current_names):
    artefacts = simulation_params[0]
    test = simulation_params[1]
    evaluator = simulation_params[2]

    ArtefactFilter.used_artefacts = ArtefactFilter.used_artefacts_all_1_2_3
    filtered_artefacts = ArtefactFilter.filtering(artefacts)
    log(ArtefactFilter.get_filtered_names())
    initial_selectors = ArtefactFilter.get_filters(current_names)
    selectors = ArtefactSelectorGenerator.select(filtered_artefacts, selection_type='SHUFFLE_ONES',
                                                 number_of_ones=3, number_of_results=1000,
                                                 initial_selectors=initial_selectors)
    print(selectors)
    results = ArtefactEvaluator.evaluate(filtered_artefacts, selectors, test, evaluator)
    ArtefactEvaluator.print_best(results)
    best_selectors = ArtefactEvaluator.get_best_n(results, 1)
    candidate_names = ArtefactFilter.get_selected_names(best_selectors[0].selector.selector)
    candidate_score = best_selectors[0].get_best_val()
    return [candidate_names], candidate_score


def annealing(init=simulation_init, move=one_phase_move):
    best_parameters, best_score, simulation_params = init()

    current_parameters, current_score = best_parameters, best_score
    t0 = 3  # todo postaviti pocetnu temperaturu
    i = 1
    end_time = datetime.datetime.now() + datetime.timedelta(hours=10)

    while datetime.datetime.now() < end_time:
        iteration_start = datetime.datetime.now()

        candidate_parameters, candidate_score = move(simulation_params, current_parameters)

        if best_score < candidate_score:
            best_parameters = candidate_parameters
            best_score = candidate_score

        # todo uzeto relativno jer ocekuje 0-1 vredgnosti
        diff = (candidate_score - current_score) / max(candidate_score, current_score)

        t = t0 / float(i)
        i = i + 1

        p_diff = exp(diff / t)

        if diff > 0 or rand() < p_diff:
            current_parameters = candidate_parameters
            current_score = candidate_score

        iteration_end = datetime.datetime.now()
        iteration_duration = iteration_end - iteration_start

        log('{}, {}, {}\n'.format(iteration_duration, best_parameters, best_score), './results/logBS')
    return


annealing()
