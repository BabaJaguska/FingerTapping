import ConversionsGenerator
import Evaluator
import MLModelTopology
import ModelTopologyGenerator
import Parameters
import Signal
import Test
import TestGenerator
import time


def main():
    starting_time = time.time()
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
    ending_time = time.time()
    
    print('Program executed in: {} min'.format(round((ending_time - starting_time)/60), 2))

    return


main()
