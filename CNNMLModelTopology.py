import MachineLearningModel
from ConfigurableMLModelTopology import ConfigurableMLModelTopology


class CNNMLModelTopology(ConfigurableMLModelTopology):
    def __init__(self, configuration):
        super().__init__(configuration)

        self.model = None
        self.init_operation = MachineLearningModel.CNNModel
        self.fit_operation = MachineLearningModel.fit_model
        self.callbacks = MachineLearningModel.def_callbacks1

        self.optimizer = MachineLearningModel.get_optimizer_adam
        self.compile_model = MachineLearningModel.compile_model

        self.to_string_formatter = 'CNNMLModel [{}]'
        self.model_name = 'CNN' + str(self.batch_size) + 'Batch' + str(self.nConvLayers) + 'ConvLayers' + str(
            self.kernel_size) + 'KERNEL' + str(self.nUnits) + 'DenseUnits' + str(
            self.initialFilters) + 'initFilt' + str(
            self.stride) + 'stride' + '{:.2f}'.format(self.dropout_rate1) + 'df' + '{:.2f}'.format(
            self.dropout_rate2) + 'ds'

        return
