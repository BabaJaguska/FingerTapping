import numpy as np

import MachineLearningModel
from ConfigurableMLModelTopology import ConfigurableMLModelTopology


class Sequential2DMLModelTopology(ConfigurableMLModelTopology):
    def __init__(self, configuration):
        super().__init__(configuration)

        self.init_operation = MachineLearningModel.CNNSequential2DModel
        self.fit_operation = MachineLearningModel.fit_model
        self.callbacks = MachineLearningModel.def_callbacks3

        self.optimizer = MachineLearningModel.get_optimizer_adam
        self.compile_model = MachineLearningModel.compile_model

        self.to_string_formatter = 'Sequential2DMLModel [{} {} {} {}]'
        self.model_name = 'Sequential2d' + str(self.initialFilters) + 'Filters' + str(
            self.kernel_size) + 'Kernels' + str(self.stride) + 'Stride' + '{:.2f}'.format(self.dropout_rate1)
        return

    def __str__(self):
        result = self.to_string_formatter.format(self.initialFilters, self.kernel_size, self.stride, self.dropout_rate1)
        return result

    def get_x(self, test_data_x):
        sizes_x = [len(test_data_x)]
        for i in range(len(test_data_x[0].shape)):
            sizes_x.append(test_data_x[0].shape[i])
        sizes_x.append(1)
        sizes_x = tuple(sizes_x)

        result = np.reshape(test_data_x, sizes_x)
        return result
