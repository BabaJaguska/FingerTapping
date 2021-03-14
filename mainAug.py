# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 13:26:15 2021

@author: minja
"""

import ConversionsGenerator
import Evaluator
import MLModelTopology
import ModelTopologyGenerator
import Parameters
import Signal
import Test
import TestGenerator
import numpy as np
import dataAugmentation
#%%


def mainAug():
    measurements = Signal.load_all(Parameters.default_root_path)
    Signal.show_signals_info(measurements, plot=Parameters.show_all)

    conversions = ConversionsGenerator.create_conversions()
    ConversionsGenerator.show_signals_info(conversions, plot=Parameters.show_all)

    tests = TestGenerator.create_tests(measurements)
    Test.show_tests_info(tests, measurements, plot=Parameters.show_all)
    
    
    # RACUNA DA SI STAVILA SAMO 1 TEST i da je on basic bez tapova i beztransforma verovatno
    converted_test = TestGenerator.convert_test(tests[0], conversions)    
    
    trainDataX, trainDataY = converted_test.train_data
    
    # trainDataX = np.expand_dims(trainDataX, 1)
    trainDataX = np.swapaxes(trainDataX, 1,2)
    # trainDataX = np.swapaxes(trainDataX, 2,3)
    
    gen, disc = dataAugmentation.train_generator(trainDataX, trainDataY)
    #augTrainX, augTrainY = getAugmentSet(tempTrainData)
    # models = ModelTopologyGenerator.create_models()
    # MLModelTopology.show_models_info(models, plot=Parameters.show_all)

    # evaluation_results = Evaluator.multiple_evaluations(tests, models, conversions)
    # Evaluator.show_evaluation_results_info(evaluation_results, plot=1)

    return


mainAug()