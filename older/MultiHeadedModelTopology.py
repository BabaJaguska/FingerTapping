import MachineLearningModel
from ConfigurableMLModelTopology import ConfigurableMLModelTopology


class MultiHeadedModelTopology(ConfigurableMLModelTopology):
    def __init__(self, configuration):
        super().__init__(configuration)

        self.model = None
        self.init_operation = MachineLearningModel.MultiHeadedModel
        self.fit_operation = MachineLearningModel.fit_model  # MachineLearningModel.fit_model_no_validation
        self.callbacks = MachineLearningModel.def_callbacks3

        self.optimizer = MachineLearningModel.get_optimizer_adam
        self.compile_model = MachineLearningModel.compile_model

        self.to_string_formatter = 'MultiHeadedMLModel [{}]'
        self.model_name = 'MultiHeaded' + str(self.batch_size) + 'Batch' + str(self.nConvLayers) + 'ConvLayers' + str(
            self.kernel_size) + 'KERNEL' + str(self.nUnits) + 'DenseUnits' + str(
            self.initialFilters) + 'initFilt' + '{:.2f}'.format(self.dropout_rate1) + 'df' + '{:.2f}'.format(
            self.dropout_rate2) + 'ds'

        return

    def get_x(self, test_data_x):
        result = []
        for i in range(self.nConvLayers):
            result.append(test_data_x)
        return result
