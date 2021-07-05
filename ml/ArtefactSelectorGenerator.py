from ml.Artefact import Artefact
from math import comb
import random


def select(artefacts, selection_type=None, number_of_ones=0, number_of_results=None, initial_selectors=None,
           start_index=None, end_index=None):
    if artefacts is not None and len(artefacts) > 0:
        number_of_artefacts = len(artefacts[0].values)
    else:
        number_of_artefacts = 0

    result = None
    if selection_type == 'FIX_NUM_OF_ONES':
        result = fix_num_of_ones(number_of_artefacts, number_of_ones,
                                 number_of_results, start_index, end_index)
    elif selection_type == 'UP_TO_NUM_OF_ONES':
        result = up_to_num_of_ones(number_of_artefacts, number_of_ones,
                                   number_of_results, start_index, end_index)
    elif selection_type == 'ADD_FIX_NUM_OF_ONES':
        result = add_fix_num_of_ones(number_of_artefacts, number_of_ones,
                                     number_of_results, initial_selectors, start_index, end_index)
    elif selection_type == 'ADD_UP_TO_NUM_OF_ONES':
        result = add_up_to_num_of_ones(number_of_artefacts, number_of_ones,
                                       number_of_results, initial_selectors, start_index, end_index)
    elif selection_type == 'CHANGE_FIX_NUM_OF_ONES':
        result = change_fix_num_of_ones(number_of_artefacts, number_of_ones,
                                        number_of_results, initial_selectors, start_index, end_index)
    elif selection_type == 'CHANGE_UP_TO_NUM_OF_ONES':
        result = change_up_to_num_of_ones(number_of_artefacts, number_of_ones,
                                          number_of_results, initial_selectors, start_index, end_index)
    elif selection_type == 'RANDOM':
        result = random_artefacts(number_of_artefacts, number_of_results)
    elif selection_type == 'SEQUENCE':
        result = sequence_artefacts(number_of_artefacts, start_index, end_index)
    elif selection_type == "REPETITION":
        result = repetition_artefacts(initial_selectors, start_index, end_index)
    else:
        result = get_all(number_of_artefacts)
    return result


def fix_num_of_ones(number_of_artefacts, number_of_ones, number_of_results, start_index=None, end_index=None):
    results = []
    number_of_ones = min(number_of_ones, number_of_artefacts)
    max_number = comb(number_of_artefacts, number_of_ones)
    if (number_of_results is None or max_number <= number_of_results) and number_of_ones < 7:
        # TODO ispraviti ovo da bude jednostavno jedinstveno resenje
        if number_of_ones == 0:
            selector = 0
            artefact_selector = ArtefactSelector(selector)
            results.append(artefact_selector)
        elif number_of_ones == 1:
            for i1 in range(number_of_artefacts):
                selector = 1 << i1
                artefact_selector = ArtefactSelector(selector)
                results.append(artefact_selector)
        elif number_of_ones == 2:
            for i1 in range(number_of_artefacts):
                for i2 in range(i1 + 1, number_of_artefacts):
                    selector = 1 << i1 | 1 << i2
                    artefact_selector = ArtefactSelector(selector)
                    results.append(artefact_selector)
        elif number_of_ones == 3:
            for i1 in range(number_of_artefacts):
                for i2 in range(i1 + 1, number_of_artefacts):
                    for i3 in range(i2 + 1, number_of_artefacts):
                        selector = 1 << i1 | 1 << i2 | 1 << i3
                        artefact_selector = ArtefactSelector(selector)
                        results.append(artefact_selector)
        elif number_of_ones == 4:
            for i1 in range(number_of_artefacts):
                for i2 in range(i1 + 1, number_of_artefacts):
                    for i3 in range(i2 + 1, number_of_artefacts):
                        for i4 in range(i3 + 1, number_of_artefacts):
                            selector = 1 << i1 | 1 << i2 | 1 << i3 | 1 << i4
                            artefact_selector = ArtefactSelector(selector)
                            results.append(artefact_selector)
        elif number_of_ones == 5:
            for i1 in range(number_of_artefacts):
                for i2 in range(i1 + 1, number_of_artefacts):
                    for i3 in range(i2 + 1, number_of_artefacts):
                        for i4 in range(i3 + 1, number_of_artefacts):
                            for i5 in range(i4 + 1, number_of_artefacts):
                                selector = 1 << i1 | 1 << i2 | 1 << i3 | 1 << i4 | 1 << i5
                                artefact_selector = ArtefactSelector(selector)
                                results.append(artefact_selector)
        elif number_of_ones == 6:
            for i1 in range(number_of_artefacts):
                for i2 in range(i1 + 1, number_of_artefacts):
                    for i3 in range(i2 + 1, number_of_artefacts):
                        for i4 in range(i3 + 1, number_of_artefacts):
                            for i5 in range(i4 + 1, number_of_artefacts):
                                for i6 in range(i5 + 1, number_of_artefacts):
                                    selector = 1 << i1 | 1 << i2 | 1 << i3 | 1 << i4 | 1 << i5 | 1 << i6
                                    artefact_selector = ArtefactSelector(selector)
                                    results.append(artefact_selector)

    else:  # racunaj neke
        used = set()
        array = []
        for i in range(number_of_artefacts):
            if i < number_of_ones:
                array.append(1)
            else:
                array.append(0)
        if number_of_results is None or number_of_results > max_number: number_of_results = max_number

        while len(used) < number_of_results:
            random.shuffle(array)
            selector = 0
            for i in range(number_of_artefacts):
                val = array[i]
                if val == 1:
                    selector = selector | 1 << i
            if selector not in used:
                artefact_selector = ArtefactSelector(selector)
                results.append(artefact_selector)
                used.add(selector)

    start_index = 0 if start_index is None else start_index
    end_index = len(results) if end_index is None else end_index

    results = results[start_index:end_index]

    return results


def up_to_num_of_ones(number_of_artefacts, number_of_ones, number_of_results, start_index, end_index):
    results = []
    for i in range(1, number_of_ones):
        sub_results = fix_num_of_ones(number_of_artefacts, i, number_of_results)
        for sub_result in sub_results:
            results.append(sub_result)

    start_index = 0 if start_index is None else start_index
    end_index = len(results) if end_index is None else end_index
    results = results[start_index:end_index]

    return results


def add_fix_num_of_ones(number_of_artefacts, number_of_ones, number_of_results, start, start_index, end_index):
    start_set = fix_num_of_ones(number_of_artefacts, number_of_ones, number_of_results, start_index, end_index)

    if start is None: start = [0]

    results = []
    vals = set()

    for s in start:
        for artefact_selector in start_set:
            selector = artefact_selector.selector | s
            if selector not in vals:
                vals.add(selector)
                selector = ArtefactSelector(selector)
                results.append(selector)

    return results


def add_up_to_num_of_ones(number_of_artefacts, number_of_ones, number_of_results, start, start_index, end_index):
    start_set = up_to_num_of_ones(number_of_artefacts, number_of_ones, number_of_results, start_index, end_index)
    start_set.append(ArtefactSelector(0))

    if start is None: start = [0]

    results = []
    vals = set()

    for s in start:
        for artefact_selector in start_set:
            selector = artefact_selector.selector | s
            if selector not in vals:
                vals.add(selector)
                selector = ArtefactSelector(selector)
                results.append(selector)

    return results


def change_fix_num_of_ones(number_of_artefacts, number_of_ones, number_of_results, start, start_index, end_index):
    start_set = fix_num_of_ones(number_of_artefacts, number_of_ones, number_of_results, start_index, end_index)
    start_set.append(ArtefactSelector(0))

    if start is None: start = [0]

    results = []
    vals = set()

    for s in start:
        for artefact_selector in start_set:
            selector = artefact_selector.selector ^ s
            if selector not in vals and selector != 0:
                vals.add(selector)
                selector = ArtefactSelector(selector)
                results.append(selector)

    return results


def change_up_to_num_of_ones(number_of_artefacts, number_of_ones, number_of_results, start, start_index, end_index):
    start_set = up_to_num_of_ones(number_of_artefacts, number_of_ones, number_of_results, start_index, end_index)
    start_set.append(ArtefactSelector(0))

    if start is None: start = [0]

    results = []
    vals = set()

    for s in start:
        for artefact_selector in start_set:
            selector = artefact_selector.selector ^ s
            if selector not in vals and selector != 0:
                vals.add(selector)
                selector = ArtefactSelector(selector)
                results.append(selector)

    return results


def random_artefacts(number_of_artefacts, number_of_results):
    results = []
    vals = set()
    max_val = (1 << number_of_artefacts) - 1

    for i in range(number_of_results):
        selector = random.randint(1, max_val)
        if selector not in vals and selector != 0:
            artefact_selector = ArtefactSelector(selector)
            results.append(artefact_selector)
            vals.add(selector)

    return results


def get_all(number_of_artefacts):
    results = []
    max_val = (1 << number_of_artefacts) - 1

    selector = max_val
    artefact_selector = ArtefactSelector(selector)
    results.append(artefact_selector)

    return results


def sequence_artefacts(number_of_artefacts, start_index, end_index):
    results = []
    vals = set()
    max_val = (1 << number_of_artefacts) - 1

    for i in range(start_index, end_index):
        selector = i & max_val
        if selector not in vals and selector != 0:
            artefact_selector = ArtefactSelector(selector)
            results.append(artefact_selector)
            vals.add(selector)

    return results


def repetition_artefacts(start, start_index, end_index):
    results = []

    for s in start:
        for i in range(start_index, end_index):
            selector = ArtefactSelector(s)
            results.append(selector)

    return results


class ArtefactSelector:
    def __init__(self, selector):
        self.selector = selector

    def __str__(self):
        temp = '{0:b}'.format(self.selector)
        return str(temp)

    def __repr__(self):
        return str(self)

    def select(self, artefacts):
        results = []
        for artefact in artefacts:
            result = self.select_for_one(artefact)
            results.append(result)
        return results

    def select_for_one(self, artefact):
        selected_values = []

        current = self.selector
        for i in range(len(artefact.values)):
            value = artefact.values[i]
            if current & 1 == 1:
                selected_values.append(value)
            current = current >> 1
        result = Artefact(artefact.name, artefact.description + '_' + artefact.result, selected_values, None,
                          artefact.result)
        return result
