import Evaluator
import MLModelTopology
import ModelTopologyGenerator
import Parameters
import Signal
import Test
import TestGenerator


def main():
    signals = Signal.load_all(Parameters.default_root_path)
    Signal.show_signals_info(signals, plot=Parameters.show_all)

    tests = TestGenerator.create_tests(signals)
    Test.show_tests_info(tests, signals, plot=Parameters.show_all)

    models = ModelTopologyGenerator.create_models()
    MLModelTopology.show_models_info(models, plot=Parameters.show_all)

    evaluation_results = Evaluator.multiple_evaluations(tests, models)
    Evaluator.show_evaluation_results_info(evaluation_results, plot=1)
    return


main()
