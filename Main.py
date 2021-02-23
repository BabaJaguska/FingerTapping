import ConversionsGenerator
import Evaluator
import MLModelTopology
import ModelTopologyGenerator
import Parameters
import Signal
import Test
import TestGenerator


def main():
    measurements = Signal.load_all(Parameters.default_root_path)
    Signal.show_signals_info(measurements, plot=Parameters.show_all)

    conversions = ConversionsGenerator.create_conversions()
    ConversionsGenerator.show_signals_info(conversions, plot=Parameters.show_all)

    tests = TestGenerator.create_tests(measurements)
    Test.show_tests_info(tests, measurements, plot=Parameters.show_all)

    models = ModelTopologyGenerator.create_models()
    MLModelTopology.show_models_info(models, plot=Parameters.show_all)

    evaluation_results = Evaluator.multiple_evaluations(tests, models, conversions)
    Evaluator.show_evaluation_results_info(evaluation_results, plot=1)

    return


main()
