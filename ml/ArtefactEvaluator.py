from datetime import datetime

import Parameters
import ArtefactFilter


def evaluate(artefacts, selectors, test, evaluator):
    results = []

    for s in selectors:
        selected_artefacts = s.select(artefacts)

        evaluation_results = []
        testing_combinations = test.create_combinations(selected_artefacts)
        for testing_combination in testing_combinations:
            single_evaluation = evaluator.evaluate(testing_combination)
            evaluation_results.append(single_evaluation)

        selectors_result = evaluator.combine_evaluations(evaluation_results)
        selectors_result.set_selector(s)
        results.append(selectors_result)
        log(selectors_result)
        if len(results) % 500 == 0: print_best(results)

    return results


def print_best(results):
    print_best_n(results)
    print_best_one(results)


def print_best_one(results):
    best_val = 0
    best_selectors = []
    last_in = None
    last_matrix = None
    for result in results:
        for key in result.results:
            val = result.results[key][0]
            if val == best_val:
                if last_in != result:
                    best_selectors.append(result)
                    last_in = result
            elif val > best_val:
                best_val = val
                best_selectors.clear()
                best_selectors.append(result)
                last_in = result
                last_matrix = result.results[key][1]
            else:
                continue

    print('---------Best results--------')
    print(last_matrix)
    log(best_selectors, './results/resultML')

    return


def print_best_n(results, n=10):
    results.sort(key=sorting_index, reverse=True)
    print('---------Best N={} results--------'.format(n))
    best_selectors = results[0:n]
    best_selectors.reverse()
    log(best_selectors, './results/n_resultML')
    return


def sorting_index(result):
    best_val = 0
    for key in result.results:
        val = result.results[key][0]
        if val > best_val:
            best_val = val
    selector = result.selector.selector
    cnt = 0
    while selector != 0:
        if selector & 1 == 1: cnt = cnt + 1
        selector = selector // 2

    return [best_val, cnt]


def log(result, file_name='./results/logML'):
    print(result)

    suffix, first = get_time_suffix()
    file_name = file_name + "_" + suffix + ".txt"

    with open(file_name, 'a') as file:
        if first: file.write(ArtefactFilter.get_filtered_names())
        file.write(str(result))
        file.close()

    return


def get_time_suffix():
    first = False
    if Parameters.file_suffix is None:
        Parameters.file_suffix = datetime.now()
        Parameters.file_suffix = Parameters.file_suffix.strftime("%Y-%m-%d_%H-%M-%S")
        first = True
    return Parameters.file_suffix, first
