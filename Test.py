import Diagnosis
import Signal


class Test:
    def __init__(self, test_type, train_data=None, validation_data=None, test_data=None,
                 measurements=None, start_time=0, end_time=0, train_percent=0, test_percent=0,
                 is_loaded=True, n_folds = 5):
        self.test_type = test_type
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.measurements = measurements
        self.start_time = start_time
        self.end_time = end_time
        self.train_percent = train_percent
        self.test_percent = test_percent
        self.is_loaded = is_loaded
        self.n_folds = n_folds

        return

    def __str__(self):
        if self.is_loaded or ((self.train_data is not None) and (self.validation_data is not None) and (
                self.test_data is not None)):
            result = '{}[train: {}, validation: {}, test: {}]'.format(self.test_type, len(self.train_data),
                                                                      len(self.validation_data),
                                                                      len(self.test_data))
        else:
            result = 'NL {}[train: {:.2f}, validation: {:.2f}, test: {:.2f}]'. \
                format(self.test_type, self.train_percent,
                       1 - self.train_percent - self.test_percent,
                       self.test_percent)

        return result

    def __repr__(self):
        return str(self)


def show_tests_info(tests, signals, plot=1):
    # plot
    if plot == 1:
        Signal.plot_class_distribution(signals)

    for test in tests:
        show_test_info(test, plot)
    return


def show_test_info(test, plot=1):
    train_data = test.train_data
    validation_data = test.validation_data
    test_data = test.test_data

    print(str(test))

    if test.is_loaded:
        show_class_distribution(train_data, 'Train', plot)
        show_class_distribution(validation_data, 'Validation', plot)
        show_class_distribution(test_data, 'Test', plot)
    return


def show_class_distribution(test, test_type, plot=1):
    diagnoses = test[1]
    Diagnosis.show_class_distribution(diagnoses, test_type, plot)

    return
