import sys

import numpy as np
from sklearn.metrics import confusion_matrix

import ConfigurationGenerator
import Diagnosis


class MLModelTopology:
    def __init__(self, configuration):
        self.configuration = configuration

        self.model = None
        self.init_operation = None
        self.fit_operation = None
        self.callbacks = None
        self.optimizer = None
        self.compile_model = None

        self.to_string_formatter = 'MLModel [{}]'
        self.model_name = 'MLModel'  # pod ovim imenom ce model biti sacuvan

        return

    def init_model(self, data_x, data_y):
        pass

    def train(self, train_data_x, train_data_y, validation_data_x, validation_data_y, epochs):
        pass

    def evaluate(self, test_data_x, test_data_y):
        pass

    def clear(self):
        self.model = None
        return

    def get_name(self):
        return self.model_name

    def show_model_info(self, plot=1):
        self.model.summary()
        return

    def save(self, path, extension='CEO.h5'):
        name = path + self.get_name() + extension
        try:
            self.model.save(name)
        except:
            print("An exception occurred during saving:" + name, sys.exc_info())
            # TODO javlja se ova greska OSError: Unable to create link (name already exists)
        return

    def calc_confusion_matrix(self, test_data_x, test_data_y):
        test_x = test_data_x
        test_y = test_data_y
        p, a = [], []
        for X, Y in zip(test_x, test_y):
            temp = self.predict_signal(X, Y, self.model, verbose=0)
            p.append(temp['Predicted'])
            a.append(temp['Actual'])

        conf_mat = confusion_matrix(a, p)
        print(conf_mat)

        return conf_mat

    def predict_signal(self, signal_x, signal_y, model, verbose=1):
        signal_x = np.expand_dims(signal_x, axis=0)
        signal_x = self.get_x(signal_x)
        prediction = model.predict(signal_x)
        prediction = prediction[0]
        prediction = np.round(np.float32(prediction), 2)

        d = Diagnosis.get_diagnosis_names_plot()

        actual_diagnosis = d[np.argmax(signal_y)]
        predicted_diagnosis = d[np.argmax(prediction)]

        if verbose:
            print('predictedDiagnosis: ', predicted_diagnosis)
            print('actualDiagnosis: ', actual_diagnosis)

            diagnosis = 'Certainty: \n'
            for i in range(len(prediction)):
                name = d[i]
                diagnosis = diagnosis + name + ': {} \n'.format(prediction[i])
            print(diagnosis)
            print('#################################################')

        return {"Predicted": predicted_diagnosis, "Actual": actual_diagnosis}

    def get_x(self, test_data_x):
        return test_data_x

    def __str__(self):
        result = self.to_string_formatter.format(ConfigurationGenerator.to_string_values(self.configuration))
        return result

    def __repr__(self):
        return str(self)


def show_models_info(models, plot=1):
    # calc

    # #print
    print('MODELS: {}'.format(len(models)))

    # plot

    return
