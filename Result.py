import itertools
import math

import matplotlib.pyplot as plt
import numpy as np

import ConfigurationGenerator
import Diagnosis
import Parameters


class Result:
    def __init__(self, model, conversions, test_type, configuration, history, test_accuracy, train_accuracy,
                 validation_accuracy,
                 avg_test_accuracy, avg_train_accuracy, avg_validation_accuracy, confuse_matrix):

        self.model = model
        self.conversions = conversions
        self.test_type = test_type
        self.configuration = configuration
        self.history = history
        self.test_accuracy = test_accuracy
        self.train_accuracy = train_accuracy
        self.validation_accuracy = validation_accuracy
        self.avg_test_accuracy = avg_test_accuracy
        self.avg_train_accuracy = avg_train_accuracy
        self.avg_validation_accuracy = avg_validation_accuracy
        self.confuse_matrix = confuse_matrix

        return

    def __str__(self):
        try:
            decimal_places = Parameters.decimal_places if Parameters.decimal_places > 2 else 2

            accuracy, precision, recall, f1 = calc_metrics(self.confuse_matrix)
            formatter = '----------\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{:.' + str(decimal_places) + 'f}\t{:.' \
                        + str(decimal_places) + 'f}\t{:.' + str(decimal_places) + 'f}\n{:.' \
                        + str(decimal_places) + 'f}\t{}\t{}\t{}\n{}\n----------\n'

            res = formatter.format(
                str(self.model),
                str(self.conversions),
                self.test_type,
                ConfigurationGenerator.to_string(self.configuration),
                clean(self.train_accuracy, decimal_places),
                clean(self.validation_accuracy, decimal_places),
                clean(self.test_accuracy, decimal_places),
                self.avg_train_accuracy,
                self.avg_validation_accuracy,
                self.avg_test_accuracy,
                accuracy, clean(precision, decimal_places - 2), clean(recall, decimal_places - 2),
                clean(f1, decimal_places - 2),
                self.confuse_matrix)
            

        except:
            res = ''
        return res

    def __repr__(self):
        return str(self)

    def save(self, file_name):
        with open(file_name, 'a') as file:
            file.write(str(self))
            file.close()
        return


def show_confuse_matrix(cm, comment='', plot=1):
    # plot
    if plot == 1:
        plot_confuse_matrix(cm, comment)
    return


def plot_confuse_matrix(cm, comment=''):
    confuse_matrix = calc_confuse_matrix_percent(cm)

    plt.figure(figsize=(10, 10))
    plt.imshow(confuse_matrix)
    plt.colorbar()
    number_of_results = Diagnosis.get_diagnosis_number()
    for i, j in itertools.product(range(number_of_results), range(number_of_results)):
        plt.text(i, j, '{}%\n({})'.format(round(confuse_matrix[j, i], 2), cm[j, i], 'd'),
                 horizontalalignment='center',
                 color='white' if confuse_matrix[j, i] < 60 else 'black',
                 size=15)
    tick_marks = np.arange(number_of_results)
    classes = Diagnosis.get_diagnosis_names()
    plt.xticks(tick_marks, classes, rotation=45, size=15)
    plt.yticks(tick_marks, classes, size=15)
    plt.ylabel('True label', size=15)
    plt.xlabel('\nPredicted label', size=15)
    plt.style.use(['tableau-colorblind10'])
    plt.title('CM: {}\n'.format(comment), size=17)
    plt.show()
    return


def calc_confuse_matrix_percent(confuse_matrix):
    sumaPoRedovima = confuse_matrix.astype('float').sum(axis=1)
    confMatPerc = [gore / dole for gore, dole in zip(confuse_matrix, sumaPoRedovima)]
    confMatPerc = np.matrix(confMatPerc) * 100
    return confMatPerc


def calc_metrics(confuse_matrix):
    all_of_class = np.sum(confuse_matrix, axis=1)
    all_as_class = np.sum(confuse_matrix, axis=0)
    correctly = np.diag(confuse_matrix)
    precision = np.divide(correctly, all_as_class) * 100
    recall = np.divide(correctly, all_of_class) * 100
    f1 = 2 * (precision * recall) / (precision + recall)
    all_all = np.sum(confuse_matrix)
    all_correct = np.sum(correctly)
    accuracy = np.round(all_correct * 100 / all_all, 2)
    return accuracy, precision, recall, f1


def clean(data, places=2):
    result = [x if not math.isnan(x) else 0.0 for x in data]
    result = np.round(result, places)
    return result


def examine_history(history):
    val_accuracy = max(history['val_accuracy']) if 'val_accuracy' in history else 0
    accuracy = max(history['accuracy']) if 'accuracy' in history else 0
    return val_accuracy, accuracy


def show_history(history, model_name, plot=1):
    # print
    val_accuracy, accuracy = examine_history(history)
    print("Max val accuracy: ", val_accuracy)
    print("Max train accuracy: ", accuracy)

    # plot
    if plot == 1:

        # plot Accuracy over Epochs
        if 'accuracy' in history:
            plt.plot(history['accuracy'])
            plt.plot(history['val_accuracy'])
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend(['Train Acc', 'Val Acc'])
            plt.title('Accuracy for {} over epochs'.format(model_name))
            plt.show()

        # plot Loss over Epochs
        if 'val_accuracy' in history:
            plt.plot(history['loss'])
            plt.plot(history['val_loss'])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(['Train Loss', 'Val Loss'])
            plt.title('Loss for {} over epochs'.format(model_name))
            plt.show()

    return


def show_max_accuracy(results, plot=1):
    max_accuracy = 0
    max_result = None
    cm = None

    for result in results:
        confuse_matrix = result.confuse_matrix
        accuracy, precision, recall, f1 = calc_metrics(confuse_matrix)
        if max_accuracy < accuracy:
            max_result = result
            max_accuracy = accuracy
            cm = result.confuse_matrix

    print('\n---MAX TEST ACCURACY:{}---\n'.format(max_accuracy))
    print(str(max_result))
    print('\n-------------\n')

    # plot
    if plot == 1:
        if max_result is not None:
            plot_confuse_matrix(cm, 'Overall ' + str(max_result.model))
        else:
            print('!!!!No confuse matrix!!!\n')
    return
