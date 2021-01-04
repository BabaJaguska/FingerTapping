import ConfigurationGenerator
import Parameters
from CNNMLModelTopology import CNNMLModelTopology
from LSTMMLModelTopology import LSTMMLModelTopology
from MultiHeadedModelTopology import MultiHeadedModelTopology
from RandomMLModelTopology import RandomMLModelTopology
from SequentialMLModel import SequentialMLModelTopology


def create_models(model_type=Parameters.model_topology_type,
                  configuration_type=Parameters.configuration_type):
    configurations = ConfigurationGenerator.create_configurations(configuration_type)

    models = []
    for configuration in configurations:
        model = None
        if model_type == 'CNNMLModel': model = CNNMLModelTopology(configuration)
        if model_type == 'CNNSequentialMLModel': model = SequentialMLModelTopology(configuration)
        if model_type == 'CNNRandomMLModel': model = RandomMLModelTopology(configuration)
        if model_type == 'LSTMMLModel': model = LSTMMLModelTopology(configuration)
        if model_type == 'MultiHeadedMLModel': model = MultiHeadedModelTopology(configuration)

        models.append(model)
    return models
