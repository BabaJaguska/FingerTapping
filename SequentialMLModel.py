import MachineLearningModel
from ConfigurableMLModelTopology import ConfigurableMLModelTopology


class SequentialMLModelTopology(ConfigurableMLModelTopology):
    def __init__(self, configuration):
        super().__init__(configuration)

        self.init_operation = MachineLearningModel.CNNSequentialModel
        self.fit_operation = MachineLearningModel.fit_model
        self.callbacks = MachineLearningModel.def_callbacks3

        self.optimizer = MachineLearningModel.get_optimizer_adam
        self.compile_model = MachineLearningModel.compile_model

        self.to_string_formatter = 'SequentialMLModel [{} {} {}]'
        self.model_name = 'Sequential' + str(self.initialFilters) + 'Filters' + str(
            self.kernel_size) + 'Kernels' + '{:.2f}'.format(self.dropout_rate1)
        return

    def __str__(self):
        result = self.to_string_formatter.format(self.initialFilters, self.kernel_size, self.dropout_rate1)
        return result
