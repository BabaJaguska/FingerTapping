import Parameters
import Signal
import ArtefactExtractor, ArtefactSelectorGenerator, \
    ArtefactEvaluatorGenerator, ArtefactEvaluator, \
    ArtefactTestGenerator, ArtefactFilter, ArtefactNormalisator


def main():
    measurements = Signal.load_all(Parameters.default_root_path)
    Signal.show_signals_info(measurements, plot=Parameters.show_all)

    artefacts = ArtefactExtractor.extract(measurements)
    print(len(artefacts))

    artefacts = ArtefactNormalisator.normalise(artefacts)
    print(len(artefacts))

    artefacts = ArtefactFilter.filtering(artefacts)
    print(ArtefactFilter.get_filtered_names())

    selectors = ArtefactSelectorGenerator.select(artefacts, selection_type='SEQUENCE',
                                                 start_index=63, end_index=64,
                                                 initial_selectors=[0b111111])
    
    hh = {}
    for a in artefacts:
        hh[a.result] = hh.get(a.result, 0)
        hh[a.result]+=1
    
    print(hh)
        
        
    print(selectors)

    test = ArtefactTestGenerator.generate()
    print(test)

    evaluator = ArtefactEvaluatorGenerator.get_evaluator()
    print(evaluator)

    results = ArtefactEvaluator.evaluate(artefacts, selectors, test, evaluator)
    ArtefactEvaluator.print_best(results)

    return


main()
