import numpy as np

import Diagnosis
import Parameters
import Signal2Test
from Test import Test


def create_tests(signals, test_type=Parameters.test_type, number_of_tests=Parameters.number_of_tests,
                 start_time=Parameters.start_time, end_time=Parameters.end_time,
                 train_percent=Parameters.train_percent,
                 test_percent=Parameters.test_percent, combinations=Parameters.combinations,
                 load=Parameters.load_all):
    tests = []
    for i in range(number_of_tests):
        test = create_test_info(signals, test_type, start_time, end_time, train_percent, test_percent, combinations)
        if load:
            test = create_test(test)
            test = convert_test(test)
        tests.append(test)

    return tests


def create_test_info(signals, test_type, start_time, end_time, train_percent, test_percent, combinations):
    test = Test(signals=signals,
                start_time=start_time,
                end_time=end_time,
                test_type=test_type,
                combinations=combinations,
                train_percent=train_percent,
                test_percent=test_percent,
                is_loaded=False)

    return test


def create_test(test):
    result = None
    if test.is_loaded:
        result = test
    else:
        if test.test_type == 'create_simple_tests':
            result = create_simple_test(test.signals, test.test_type, test.combinations, test.train_percent,
                                        test.test_percent)
        if test.test_type == 'create_mixed_tests':
            result = create_mixed_test(test.signals, test.test_type, test.combinations, test.train_percent,
                                       test.test_percent)
    result.start_time = test.start_time
    result.end_time = test.end_time
    return result


def create_simple_test(signals, test_type, combinations, train_percent, test_percent):
    train_data, test_data, validation_data = [], [], []
    for diagnosis in Diagnosis.get_diagnosis_names():
        extract_test_and_concatenate(signals, diagnosis, train_percent, test_percent, train_data, test_data,
                                     validation_data)
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    np.random.shuffle(validation_data)
    test = Test(test_type=test_type,
                combinations=combinations,
                train_data=train_data,
                test_data=test_data,
                validation_data=validation_data)
    return test


def extract_test_and_concatenate(signals, diagnosis, train_percent, test_percent, train_data, test_data,
                                 validation_data):
    trds, teds, vads = extract_test(signals, diagnosis, train_percent, test_percent)
    for trd in trds:
        train_data.append(trd)
    for ted in teds:
        test_data.append(ted)
    for vad in vads:
        validation_data.append(vad)
    return


def extract_test(signals, diagnosis, train_percent, test_percent):
    candidates = [sig for i, sig in enumerate(signals) if sig.diagnosis == diagnosis]
    validation_percent = 1 - train_percent - test_percent
    train_data, test_data, validation_data = [], [], []
    # Split into train, test, val sets

    train_num = int(train_percent * len(candidates))
    test_num = int(test_percent * len(candidates))
    validation_num = int(validation_percent * len(candidates))

    while len(validation_data) <= validation_num:
        index = np.random.randint(0, len(candidates))
        initials = candidates[index].initials
        signals = [candidate for i, candidate in enumerate(candidates) if candidate.initials == initials]
        for signal in signals:
            validation_data.append(signal)
            candidates.remove(signal)

    while len(test_data) <= test_num:
        index = np.random.randint(0, len(candidates))
        initials = candidates[index].initials
        signals = [candidate for i, candidate in enumerate(candidates) if candidate.initials == initials]
        for signal in signals:
            test_data.append(signal)
            candidates.remove(signal)

    for signal in candidates:
        train_data.append(signal)

    return train_data, test_data, validation_data


def create_mixed_test(signals, test_type, combinations, train_percent,
                      test_percent):  # TODO da li je ovo ispravno?
    inds = []
    for diagnosis in Diagnosis.get_diagnosis_names():
        ind = [i for i, sig in enumerate(signals) if sig.diagnosis == diagnosis]
        inds.append([ind])

    # Split into train, test, val sets
    data_train = []
    data_test = []
    data_validation = []
    for DIAGind in inds:
        diag_train, diag_test, diag_val = split_one_diagnosis(signals, DIAGind[0], train_percent, test_percent)
        data_train = data_train + diag_train
        data_test = data_test + diag_test
        data_validation = data_validation + diag_val

    test = Test(test_type=test_type,
                combinations=combinations,
                train_data=data_train,
                test_data=data_test,
                validation_data=data_validation)
    return test


def split_one_diagnosis(signals, DIAGind, trainPercent, testPercent):
    # np.random.seed(12345) # TODO ako se ovo stavi uvek vraca istu raspodelu!!
    trainInd = np.random.choice(DIAGind, round(trainPercent * len(DIAGind)), replace=False)
    Xtrain = [signals[i] for i in trainInd]
    leftover = [i for i in DIAGind if i not in trainInd]

    testInd = np.random.choice(leftover, round(testPercent * len(DIAGind)), replace=False)
    Xtest = [signals[i] for i in testInd]
    leftover = [i for i in leftover if i not in testInd]

    Xval = [signals[i] for i in leftover]

    return Xtrain, Xtest, Xval


def convert_test(test):
    train_data = test.train_data
    test_data = test.test_data
    validation_data = test.validation_data
    start_time = test.start_time
    end_time = test.end_time
    combinations = test.combinations

    Xtrain, Ytrain = Signal2Test.convert_signals(train_data, combinations, start_time, end_time)
    Xtest, Ytest = Signal2Test.convert_signals(test_data, combinations, start_time, end_time)
    Xval, Yval = Signal2Test.convert_signals(validation_data, combinations, start_time, end_time)
    test = Test(combinations=test.combinations,
                test_type=test.test_type,
                train_data=(Xtrain, Ytrain),
                test_data=(Xtest, Ytest),
                validation_data=(Xval, Yval),
                start_time=test.start_time,
                end_time=test.end_time)
    return test


def load_test(test):
    test = create_test(test)
    test = convert_test(test)
    return test
