import MachineLearningModel
from ConfigurableMLModelTopology import ConfigurableMLModelTopology


class CNNLSTMMLModelTopology(ConfigurableMLModelTopology):
    def __init__(self, configuration):
        super().__init__(configuration)

        self.init_operation = MachineLearningModel.CNNLSTMModel
        self.fit_operation = MachineLearningModel.fit_model
        self.callbacks = MachineLearningModel.def_callbacks1

        self.optimizer = MachineLearningModel.get_optimizer_adam
        self.compile_model = MachineLearningModel.compile_model

        self.to_string_formatter = 'CNNLTSM'
        self.model_name = 'CNNLTSM'
        return
