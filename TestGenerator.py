import numpy as np
from sklearn.model_selection import train_test_split
import Diagnosis
import Parameters
import Signal2Test
from Test import Test


def create_tests(measurements, test_type=Parameters.test_type, number_of_tests=Parameters.number_of_tests,
                 start_time=Parameters.start_time, end_time=Parameters.end_time,
                 train_percent=Parameters.train_percent,
                 test_percent=Parameters.test_percent):
    tests = []
    for i in range(number_of_tests):
        test = create_test_info(measurements, test_type, start_time, end_time, train_percent, test_percent)
        test = create_test(test)
        tests.append(test)

    return tests


def create_test_info(measurements, test_type, start_time, end_time, train_percent, test_percent):
    test = Test(measurements=measurements,
                start_time=start_time,
                end_time=end_time,
                test_type=test_type,
                train_percent=train_percent,
                test_percent=test_percent,
                is_loaded=False)

    return test


def create_test(test):
    result = None
    if test.is_loaded:
        result = test
    elif test.test_type == 'create_simple_tests':
        result = create_simple_test(test.measurements, test.test_type, test.train_percent,
                                    test.test_percent)
    elif test.test_type == 'create_mixed_tests':
        result = create_mixed_test(test.measurements, test.test_type, test.train_percent,
                                   test.test_percent)
    result.measurements = test.measurements
    result.start_time = test.start_time
    result.end_time = test.end_time
    return result


def create_simple_test(measurements, test_type, train_percent, test_percent):
    train_data, test_data, validation_data = [], [], []
    for diagnosis in Diagnosis.get_diagnosis_names():
        extract_test_and_concatenate(measurements, diagnosis, train_percent, test_percent, train_data, test_data,
                                     validation_data)
    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    np.random.shuffle(validation_data)
    test = Test(test_type=test_type,
                train_data=train_data,
                test_data=test_data,
                validation_data=validation_data,
                is_loaded=False)
    return test


def extract_test_and_concatenate(measurements, diagnosis, train_percent, test_percent, train_data, test_data,
                                 validation_data):
    
       
    trds, teds, vads = extract_test(measurements, diagnosis, train_percent, test_percent)
    
    
    
    for trd in trds:
        train_data.append(trd)
    for ted in teds:
        test_data.append(ted)
    for vad in vads:
        validation_data.append(vad)
    return


def extract_test(measurements, diagnosis, train_percent, test_percent):
    
    # promenila da se deli po id, ne po candidate signals
    # staviti seed
    
    
    
    candidates = [sig for i, sig in enumerate(measurements) if sig.diagnosis == diagnosis]
    validation_percent = 1 - train_percent - test_percent
    train_data, test_data, validation_data = [], [], []
    # Split into train, test, val sets
    
    all_ids_in_diagnosis = np.unique([candidate.initials for candidate in candidates])
    

    train_num = int(train_percent * len(all_ids_in_diagnosis))
    test_num = int(test_percent * len(all_ids_in_diagnosis))
    validation_num = int(validation_percent * len(all_ids_in_diagnosis))
    
    np.random.seed(0); val_indices = np.random.choice(len(all_ids_in_diagnosis), validation_num)
    
    for index in val_indices:
        initials = all_ids_in_diagnosis[index]
        measurements = [candidate for i, candidate in enumerate(candidates) if candidate.initials == initials]
        for measurement in measurements:
            validation_data.append(measurement)
            candidates.remove(measurement)
            
    all_ids_in_diagnosis = np.unique([candidate.initials for candidate in candidates])
    np.random.seed(0); test_indices = np.random.choice(len(all_ids_in_diagnosis), test_num)

    for index in test_indices:
        initials = candidates[index].initials
        measurements = [candidate for i, candidate in enumerate(candidates) if candidate.initials == initials]
        for measurement in measurements:
            test_data.append(measurement)
            candidates.remove(measurement)

    for measurement in candidates:
        train_data.append(measurement)
        

    return train_data, test_data, validation_data


def create_mixed_test(measurements, test_type, train_percent, test_percent):  # TODO da li je ovo ispravno?
    inds = []
    for diagnosis in Diagnosis.get_diagnosis_names():
        ind = [i for i, sig in enumerate(measurements) if sig.diagnosis == diagnosis]
        inds.append([ind])

    # Split into train, test, val sets
    data_train = []
    data_test = []
    data_validation = []
    for DIAGind in inds:
        diag_train, diag_test, diag_val = split_one_diagnosis(measurements, DIAGind[0], train_percent, test_percent)
        data_train = data_train + diag_train
        data_test = data_test + diag_test
        data_validation = data_validation + diag_val

    test = Test(test_type=test_type,
                train_data=data_train,
                test_data=data_test,
                validation_data=data_validation,
                is_loaded=False)
    return test


def split_one_diagnosis(measurements, DIAGind, trainPercent, testPercent):
    # np.random.seed(12345) # TODO ako se ovo stavi uvek vraca istu raspodelu!!
    trainInd = np.random.choice(DIAGind, round(trainPercent * len(DIAGind)), replace=False)
    Xtrain = [measurements[i] for i in trainInd]
    leftover = [i for i in DIAGind if i not in trainInd]

    testInd = np.random.choice(leftover, round(testPercent * len(DIAGind)), replace=False)
    Xtest = [measurements[i] for i in testInd]
    leftover = [i for i in leftover if i not in testInd]

    Xval = [measurements[i] for i in leftover]

    return Xtrain, Xtest, Xval


def convert_test(test, conversions):
    train_data = test.train_data
    test_data = test.test_data
    validation_data = test.validation_data
    start_time = test.start_time
    end_time = test.end_time

    Xtrain, Ytrain = Signal2Test.convert_measurements(train_data, conversions, start_time, end_time)
    Xtest, Ytest = Signal2Test.convert_measurements(test_data, conversions, start_time, end_time)
    Xval, Yval = Signal2Test.convert_measurements(validation_data, conversions, start_time, end_time)
    test = Test(test_type=test.test_type,
                train_data=(Xtrain, Ytrain),
                test_data=(Xtest, Ytest),
                validation_data=(Xval, Yval),
                start_time=test.start_time,
                end_time=test.end_time,
                is_loaded=True)
    return test
