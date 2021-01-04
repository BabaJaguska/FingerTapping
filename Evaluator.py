import sys

import numpy as np

import Parameters
import Result
import TestGenerator


def multiple_evaluations(tests, models, path=Parameters.default_results_path,
                         result_file=Parameters.default_results_file):
    results = []

    for model in models:
        try:
            print(str(model))
            model_results = []
            for test in tests:
                test = TestGenerator.load_test(test)
                res = single_evaluation(test, model, path)
                model_results.append(res)

            res = combine_model_results(model_results)
            results.append(res)

            res.save(path + result_file)
            show_evaluation_results_info([res], plot=1)

        except:
            print("An exception occurred {}".format(str(model)), sys.exc_info())

    return results


def single_evaluation(test, model, path,
                      number_of_tries_per_configurations=Parameters.number_of_tries_per_configurations,
                      epochs=Parameters.epochs):
    validation_accuracies = []
    train_accuracies = []
    test_accuracies = []
    histories = []
    cms = []

    train_data = test.train_data
    validation_data = test.validation_data
    test_data = test.test_data

    for i in range(number_of_tries_per_configurations):
        model.init_model(train_data[0], train_data[1])

        model.show_model_info()

        model.save(path)

        history, train_accuracy, validation_accuracy = model.train(train_data[0], train_data[1], validation_data[0],
                                                                   validation_data[1], epochs)
        histories.append(history)
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)

        Result.show_history(history.history, model.get_name(), plot=Parameters.show_all)

        confuse_matrix, test_accuracy = model.evaluate(test_data[0], test_data[1])
        cms.append(confuse_matrix)
        test_accuracies.append(test_accuracy)

        Result.show_confuse_matrix(confuse_matrix, str(model) + str(i), plot=Parameters.show_all)

    res = Result.Result(model=model,
                        combinations=test.combinations,
                        test_type=test.test_type,
                        configuration=model.get_configuration(),
                        history=histories,
                        train_accuracy=train_accuracies,
                        validation_accuracy=validation_accuracies,
                        test_accuracy=test_accuracies,
                        avg_validation_accuracy=np.mean(validation_accuracies),
                        avg_train_accuracy=np.mean(train_accuracies),
                        avg_test_accuracy=np.mean(test_accuracies),
                        confuse_matrix=cms)
    return res


def combine_model_results(model_results):
    model = model_results[0].model
    configuration = model_results[0].configuration
    combinations = model_results[0].combinations

    test_type = model_results[0].test_type

    histories = []
    for configuration_result in model_results:
        for data in configuration_result.history:
            histories.append(data)

    train_accuracy = []
    for configuration_result in model_results:
        for data in configuration_result.train_accuracy:
            train_accuracy.append(data)

    validation_accuracy = []
    for configuration_result in model_results:
        for data in configuration_result.validation_accuracy:
            validation_accuracy.append(data)

    test_accuracy = []
    for configuration_result in model_results:
        for data in configuration_result.test_accuracy:
            test_accuracy.append(data)

    size = len(model_results[0].confuse_matrix[0])
    cms = np.zeros((size, size), dtype=int)
    for configuration_result in model_results:
        for cm in configuration_result.confuse_matrix:
            cms = cms + cm

    res = Result.Result(model=model,
                        combinations=combinations,
                        test_type=test_type,
                        configuration=configuration,
                        history=histories,
                        train_accuracy=train_accuracy,
                        validation_accuracy=validation_accuracy,
                        test_accuracy=test_accuracy,
                        avg_validation_accuracy=np.mean(validation_accuracy),  # TODO nije bas srednja vrednost ali lici
                        avg_train_accuracy=np.mean(train_accuracy),
                        avg_test_accuracy=np.mean(test_accuracy),
                        confuse_matrix=cms)

    return res


def show_evaluation_results_info(results, plot=1):
    print('RESULTS:')
    Result.show_max_accuracy(results, plot)

    for res in results:
        print(str(res))

    # plot

    return
