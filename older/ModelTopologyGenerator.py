import ConfigurationGenerator
import Parameters
from CNNLSTMMLModel import CNNLSTMMLModelTopology
from CNNMLModelTopology import CNNMLModelTopology
from LSTMMLModelTopology import LSTMMLModelTopology
from MultiHeadedModelTopology import MultiHeadedModelTopology
from RandomMLModelTopology import RandomMLModelTopology
from Sequential2DMLModel import Sequential2DMLModelTopology
from SequentialMLModel import SequentialMLModelTopology


def create_models(model_type=Parameters.model_topology_type,
                  configuration_type=Parameters.configuration_type):
    configurations = ConfigurationGenerator.create_configurations(configuration_type)

    models = []
    for configuration in configurations:
        model = None
        if model_type == 'CNNMLModel': model = CNNMLModelTopology(configuration)
        elif model_type == 'CNNSequentialMLModel': model = SequentialMLModelTopology(configuration)
        elif model_type == 'CNNSequential2DMLModel': model = Sequential2DMLModelTopology(configuration)
        elif model_type == 'CNNRandomMLModel': model = RandomMLModelTopology(configuration)
        elif model_type == 'LSTMMLModel': model = LSTMMLModelTopology(configuration)
        elif model_type == 'MultiHeadedMLModel': model = MultiHeadedModelTopology(configuration)
        elif model_type == 'CNNLSTMMLModel': model = CNNLSTMMLModelTopology(configuration)

        models.append(model)
    return models
