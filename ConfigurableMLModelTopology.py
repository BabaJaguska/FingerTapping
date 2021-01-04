import Result
from MLModelTopology import MLModelTopology


class ConfigurableMLModelTopology(MLModelTopology):
    def __init__(self, configuration):
        super().__init__(configuration)

        self.kernel_size = configuration['kernelSize']
        self.nConvLayers = configuration['nConvLayers']
        self.batch_size = configuration['batchSize']
        self.stride = configuration['stride']
        self.kernel_constraint = configuration['constraint']
        self.nUnits = configuration['nDenseUnits']
        self.initialFilters = configuration['nInitialFilters']
        self.dropout_rate1 = configuration['dropout_rate1']
        self.dropout_rate2 = configuration['dropout_rate2']

        return

    def init_model(self, data_x, data_y):
        self.model = self.init_operation(data_x, data_y, self.nConvLayers, self.kernel_size, self.stride,
                                         self.kernel_constraint, self.nUnits, self.initialFilters,
                                         self.dropout_rate1, self.dropout_rate2)
        # self.model.summary()

        self.compile_model(self.model, self.optimizer())

        return

    def train(self, train_data_x, train_data_y, validation_data_x, validation_data_y, epochs):
        print('TRAINING...')
        history = self.fit_operation(self.model, self.get_name(), self.callbacks, self.get_x(train_data_x),
                                     train_data_y, self.get_x(validation_data_x), validation_data_y, epochs,
                                     self.batch_size)
        validation_accuracy, train_accuracy = Result.examine_history(history.history)

        return history, train_accuracy, validation_accuracy

    def evaluate(self, test_data_x, test_data_y):
        actual = self.model.evaluate(self.get_x(test_data_x), test_data_y)
        test_accuracy = actual[1]

        confuse_matrix = self.calc_confusion_matrix(test_data_x, test_data_y)

        return confuse_matrix, test_accuracy

    def get_configuration(self):
        return self.configuration
