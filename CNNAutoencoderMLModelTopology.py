import MachineLearningModel
import Result
from ConfigurableMLModelTopology import ConfigurableMLModelTopology


class CNNAutoencoderMLModelTopology(ConfigurableMLModelTopology):
    def __init__(self, configuration):
        super().__init__(configuration)

        self.autoencoder = None
        self.encoder = None

        self.to_string_formatter = 'CNNAutoencoderMLModel [{}]'
        self.model_name = 'CNNAutoencoderShuffled' + str(self.batch_size) + 'Batch' + str(
            self.nConvLayers) + 'ConvLayers' + str(self.kernel_size) + 'KERNEL' + str(self.nUnits) + 'DenseUnits' + str(
            self.initialFilters) + str(self.stride) + 'stride' + 'initFilt' + '{:.2f}'.format(
            self.dropout_rate1) + 'df' + '{:.2f}'.format(
            self.dropout_rate2) + 'ds'

        return

    def init_model(self, data_x, data_y):
        [self.autoencoder, self.encoder] = MachineLearningModel.CNNModelAutoencoder(data_x, data_y, self.nConvLayers,
                                                                                    self.kernel_size,
                                                                                    self.stride,
                                                                                    self.kernel_constraint, self.nUnits,
                                                                                    self.initialFilters,
                                                                                    self.dropout_rate1,
                                                                                    self.dropout_rate2)
        self.autoencoder.summary()
        self.autoencoder.compile(optimizer='adam',
                                 metrics=['accuracy'],
                                 loss='binary_crossentropy')
        self.model = self.autoencoder
        return

    def train(self, train_data_x, train_data_y, validation_data_x, validation_data_y, epochs):
        # TODO ovo jos nije implementirano
        print('TRAINING...')
        history = MachineLearningModel.fit_modelA(self.autoencoder, self.get_name(), train_data_x, train_data_y,
                                                  validation_data_x, validation_data_y,
                                                  epochs, self.batch_size)
        validation_accuracy, train_accuracy = Result.examine_history(history.history)

        return history, train_accuracy, validation_accuracy

    def evaluate(self, test_data_x, test_data_y):
        actual = self.autoencoder.evaluate(test_data_x, test_data_y)
        test_accuracy = actual[1]

        confuse_matrix = self.calc_confusion_matrix(test_data_x, test_data_y)

        return confuse_matrix, test_accuracy
