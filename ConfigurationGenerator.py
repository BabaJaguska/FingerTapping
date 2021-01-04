import numpy as np

import Parameters


def create_configurations(configuration_type):
    configurations = None
    if configuration_type == 'random_attr': configurations = create_configurations_random_attr(
        Parameters.number_of_configurations, Parameters.attribute_range_values)

    if configuration_type == 'one_attr': configurations = create_configurations_one_attr(
        Parameters.one_attr_name, Parameters.one_attr_start, Parameters.one_attr_end,
        Parameters.one_attr_step, Parameters.attribute_default_values)
    return configurations


def create_configurations_random_attr(cnt, attributes_values=(), add_default=1):
    result = []
    result_tags = set()
    while len(result) < cnt:
        val = {}
        for attribute in attributes_values:
            attribute_name = attribute[0]
            # def_val = val[1]
            min_val = attribute[2]
            max_val = attribute[3]
            if isinstance(min_val, int) and isinstance(max_val, int):
                new_val = np.random.randint(min_val, max_val + 1)
            else:
                new_val = np.random.uniform(min_val, max_val)
                new_val = round(new_val, 2)
            val[attribute_name] = new_val
        val_tag = str(val)
        if val_tag not in result_tags:
            result.append(val)
            result_tags.add(val_tag)

    if add_default == 1:
        val = {}
        for attribute in attributes_values:
            attribute_name = attribute[0]
            def_val = attribute[1]
            val[attribute_name] = def_val
        val_tag = str(val)
        if val_tag not in result_tags:
            result.append(val)
            result_tags.add(val_tag)
    return result


def create_configurations_one_attr(name, min_value, max_value, step, attributes_values=()):
    result = []

    x = min_value
    while x <= max_value:
        val = {}
        for attribute in attributes_values:
            attribute_name = attribute[0]
            def_val = attribute[1]
            val[attribute_name] = def_val
        val[name] = x
        x = x + step
        result.append(val)

    return result


def to_string_values(configuration):
    keys = configuration.keys()
    keys = [x for x in keys]
    keys.sort()
    result = ''
    for key in keys:
        result = result + str(configuration[key]) + ' '

    if len(result) > 0:
        result = result[0:-1]
    return result


def to_string_keys(configuration):
    keys = configuration.keys()
    keys = [x for x in keys]
    keys.sort()
    result = ''
    for key in keys:
        result = result + str(key) + ' '

    if len(result) > 0:
        result = result[0:-1]
    return result


def to_string(configuration):
    keys = configuration.keys()
    keys = [x for x in keys]
    keys.sort()
    result = ''
    for key in keys:
        result = result + str(key) + ': ' + str(configuration[key]) + ', '

    if len(result) > 0:
        result = result[0:-2]
    return result
