import MachineLearningModel
from ConfigurableMLModelTopology import ConfigurableMLModelTopology


class RandomMLModelTopology(ConfigurableMLModelTopology):
    def __init__(self, configuration):
        super().__init__(configuration)

        self.init_operation = MachineLearningModel.CNNRandomModel
        self.fit_operation = MachineLearningModel.fit_model
        self.callbacks = MachineLearningModel.def_callbacks1

        self.optimizer = MachineLearningModel.get_optimizer_adam
        self.compile_model = MachineLearningModel.compile_model

        self.to_string_formatter = 'CNNRandomLModel ???'
        self.model_name = 'CNNRandomLModel'
        return

    def get_name(self):
        self.to_string_formatter = self.model.name + ' [{}]'
        self.model_name = self.model.name + str(self.batch_size) + 'Batch' + str(self.nConvLayers) + 'ConvLayers' + str(
            self.kernel_size) + 'KERNEL' + str(self.nUnits) + 'DenseUnits' + str(
            self.initialFilters) + str(self.stride) + 'stride' + 'initFilt' + '{:.2f}'.format(
            self.dropout_rate1) + 'df' + '{:.2f}'.format(self.dropout_rate2) + 'ds'
        return self.model_name
