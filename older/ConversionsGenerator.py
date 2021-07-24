import random

import Parameters


def create_conversions(conversions_type=Parameters.conversion_type):
    result = None
    if conversions_type == 'create_simple':
        result = create_simple_conversions()
    elif conversions_type == 'create_full_list':
        result = create_full_list_conversions()
    elif conversions_type == 'create_random_list':
        result = create_random_list_conversions()
    elif conversions_type == 'add_all_to_list':
        result = add_one_full_list_conversions()
    elif conversions_type == 'add_random_list':
        result = add_one_random_conversions()
    return result


def create_simple_conversions(conversions=Parameters.conversion_combinations):
    result = [conversions]
    return result


def create_full_list_conversions():
    result = []

    signal_types = [getattr(Parameters.SignalType, x) for x in dir(Parameters.SignalType) if not x.startswith("__")]

    tap_types = [getattr(Parameters.TapType, x) for x in dir(Parameters.TapType) if not x.startswith("__")]

    function_types = [getattr(Parameters.FunctionType, x) for x in dir(Parameters.FunctionType) if
                      not x.startswith("__")]

    concatenation_types = [getattr(Parameters.ConcatenationType, x) for x in dir(Parameters.ConcatenationType) if
                           not x.startswith("__")]

    for function_type in function_types:
        for concatenation_type in concatenation_types:
            for tap_type in tap_types:
                for signal_type in signal_types:
                    tmp = [[signal_type, tap_type, function_type, concatenation_type]]
                    result.append(tmp)
    return result


def create_random_list_conversions(cnt=Parameters.number_of_conversions):
    result = create_full_list_conversions()
    random.shuffle(result)
    result = result[0:cnt]
    return result


def add_one_full_list_conversions(conversions=Parameters.conversion_combinations):
    result = []
    full_list = create_full_list_conversions()
    for element in full_list:
        new_element = conversions.copy()
        new_element.append(element[0])
        result.append(new_element)
    return result


def add_one_random_conversions(conversions=Parameters.conversion_combinations, cnt=Parameters.number_of_conversions):
    result = []
    full_list = create_random_list_conversions(cnt)
    for element in full_list:
        new_element = conversions.copy()
        new_element.append(element[0])
        result.append(new_element)
    return result


def change_one_random_conversions(conversions=Parameters.conversion_combinations, cnt=Parameters.number_of_conversions):
    result = conversions  # TODO not implemented
    return result


def show_signals_info(conversions, plot):
    for conversion in conversions:
        print(conversion)
    return
