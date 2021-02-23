from tqdm import tqdm

import Diagnosis
import Parameters
import Signal
import TestGenerator


def validate(train_data, test_data, validation_data):
    train_set = set()
    test_set = set()
    validation_set = set()
    for tr in train_data:
        train_set.add(tr.initials)
    for te in test_data:
        test_set.add(te.initials)
    for va in validation_data:
        validation_set.add(va.initials)

    for data in train_set:
        if data in test_set: return True
        if data in validation_set: return True

    for data in test_set:
        if data in train_set: return True
        if data in validation_set: return True

    for data in validation_set:
        if data in test_set: return True
        if data in train_set: return True

    return False


def main():
    measurements = Signal.load_all(Parameters.default_root_path)
    train_percent = 0.7
    test_percent = 0.2

    for i in tqdm(range(100)):
        for diagnosis in Diagnosis.get_diagnosis_names():
            train_data, test_data, validation_data = TestGenerator.extract_test(measurements, diagnosis,
                                                                                train_percent, test_percent)
            result = validate(train_data, test_data, validation_data)
            if result: print('not ok')

    return


main()
