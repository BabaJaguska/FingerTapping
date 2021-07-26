import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier 


import Diagnosis


def get_evaluator():
    evaluator = Evaluator()
    return evaluator


class Evaluator:
    def __init__(self):
        self.models = []
        self.crete_models()
        return

    def crete_models(self):
        self.models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=1000)))
        self.models.append(('LDA', LinearDiscriminantAnalysis()))
        self.models.append(('KNN', KNeighborsClassifier()))
        self.models.append(('CART', DecisionTreeClassifier()))
        self.models.append(('NB', GaussianNB()))
        self.models.append(('SVM', SVC(gamma='auto')))
        # self.models.append(('MLP', MLPClassifier(hidden_layer_sizes = (100,50), activation = 'logistic')))

        # self.models.append(('KNN1', KNeighborsClassifier(metric='euclidean')))
        # self.models.append(('KNN2', KNeighborsClassifier(metric='manhattan')))
        # self.models.append(('KNN3', KNeighborsClassifier(metric='chebyshev')))
        # self.models.append(('KNN4', KNeighborsClassifier(metric='minkowski', p=2)))
        return

    def __str__(self):
        temp = '['
        for mode_wrapper in self.models:
            temp = temp + " " + mode_wrapper[0]
        temp = temp + ']'
        return temp

    def __repr__(self):
        return str(self)

    def evaluate(self, testing_combination):
        trains = testing_combination[0]
        tests = testing_combination[1]
        x_train, y_train = artefacts_to_nd(trains)
        x_validation, y_validation = artefacts_to_nd(tests)
        results = dict()

        # self.crete_models()

        for model_wrapper in self.models:
            model_name = model_wrapper[0]
            model = model_wrapper[1]

            # x_train, x_validation = self.transform(x_train, x_validation)

            model.fit(x_train, y_train)

            predictions = model.predict(x_validation)
            conf_mat = calc_confusion_matrix(y_validation, predictions)

            score = accuracy_score(y_validation, predictions)
            results[model_name] = (len(x_validation) * score, conf_mat)

            # print('Score: {} {} {} {} {} {}'.format(model_name, len(x_validation), score, tests[0].description, y_validation[0], predictions))

        evaluation_result = EvaluationResults(0, results)
        return evaluation_result

    def transform(self, x_train, x_test):
        pca = PCA(n_components=2)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)
        return x_train, x_test

    def combine_evaluations(self, evaluation_results):
        results_values = dict()
        result = EvaluationResults(None, results_values)
        for evaluation_result in evaluation_results:
            result.selector = evaluation_result.selector

            for key in evaluation_result.results:
                evaluation = evaluation_result.results[key]
                val = evaluation[0]
                conf_matrix = evaluation[1]

                if key in results_values:
                    old_evaluation = results_values[key]
                    old_val = old_evaluation[0]
                    old_conf_matrix = old_evaluation[1]

                    val = val + old_val
                    conf_matrix = conf_matrix + old_conf_matrix

                results_values[key] = (val, conf_matrix)

        return result


def artefacts_to_nd(artefacts):
    x = []
    y = []
    for artefact in artefacts:
        temp = []
        for xxx in artefact.values:    
            if isinstance(xxx, np.ndarray) or isinstance(xxx, list):
                for ttt in xxx:
                    temp.append(ttt)    
            else:
                temp.append(xxx)
        x.append(temp)
        y.append(artefact.result)   
    
    return x, y


def calc_confusion_matrix(validations, predictions):
    diagnoses = Diagnosis.get_diagnosis_names()
    size = len(diagnoses)
    positions = dict()
    i = 0
    for diagnosis in diagnoses:
        positions[diagnosis] = i
        i = i + 1

    conf_matrix = np.zeros((size, size))

    for validation, prediction in zip(validations, predictions):
        validation_index = positions[validation]
        prediction_index = positions[prediction]
        conf_matrix[prediction_index][validation_index] = conf_matrix[prediction_index][validation_index] + 1

    return conf_matrix


class EvaluationResults:
    def __init__(self, selector, results):
        self.selector = selector
        self.results = results

    def __str__(self):
        temp = '\n{}'.format(self.selector)

        for key in self.results.keys():
            val = self.results[key]
            temp = temp + "\t{}\t{}".format(key, val[0])

        return str(temp)

    def __repr__(self):
        return str(self)

    def set_selector(self, selector):
        self.selector = selector
        return

    def str_all(self):
        temp = '\n{}'.format(self.selector)

        for key in self.results.keys():
            val = self.results[key]
            temp = temp + "\n{}\t{}\t{}".format(key, val[0], val[1])
        return str(temp)
