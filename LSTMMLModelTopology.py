import MachineLearningModel
from ConfigurableMLModelTopology import ConfigurableMLModelTopology


class LSTMMLModelTopology(ConfigurableMLModelTopology):
    def __init__(self, configuration):
        super().__init__(configuration)

        self.model = None
        self.init_operation = MachineLearningModel.LSTMModel
        self.fit_operation = MachineLearningModel.fit_model
        self.callbacks = MachineLearningModel.def_callbacks3

        self.optimizer = MachineLearningModel.get_optimizer_adam
        self.compile_model = MachineLearningModel.compile_model

        self.to_string_formatter = 'RNNMLModel [{}]'
        self.model_name = 'RNNShuffled' + str(self.batch_size) + 'Batch' + str(
            self.nConvLayers) + 'ConvLayers' + str(self.kernel_size) + 'KERNEL' + str(
            self.nUnits) + 'DenseUnits' + str(self.initialFilters) + 'initFilt' + '{:.2f}'.format(
            self.dropout_rate1) + 'do' + '{:.2f}'.format(self.dropout_rate2) + 'do'

        return
