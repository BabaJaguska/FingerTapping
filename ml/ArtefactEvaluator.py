from datetime import datetime

import Parameters
import ArtefactFilter
from matplotlib import pyplot as plt
from itertools import product
import numpy as np


def plotConfMat(cm, cmPerc, filePathToSavePlot, showPlots=0):
    '''
    Plot a given confusion matrix with both absolute and percent values
    Then save the plot as .png

    Parameters
    ----------
        cm (): Confusion Matrix
        cmPerc (): Confusion Matrix but with percentage values (% of rows)
        filePathToSavePlot (): [file path] + filename of the image to be saved

    Returns
    --------
        None
    '''

    plt.figure(figsize=(20, 20))
    plt.imshow(cmPerc)
    plt.colorbar()
    N_classes = len(cm)

    classes = ['MSA', 'PD', 'PSP', 'HC']

    for i, j in product(range(N_classes), range(N_classes)):
        plt.text(i, j,
                 cm[j, i],
                 horizontalalignment="center",
                 color='white' if cmPerc[j, i] < 60 else "black",
                 size=35)
    tick_marks = np.arange(N_classes)

    plt.xticks(tick_marks, classes, rotation=45, size=27)
    plt.yticks(tick_marks, classes, size=27)
    plt.ylabel('True diagnosis', size=30)
    plt.xlabel('\nPredicted diagnosis', size=30)
    #  plt.style.use(['tableau-colorblind10'])
    #  plt.rcparams.image.cmap'] = 'viridis'
    plt.title('Confusion matrix\n', size=32)
    # fig1 = plt.gcf()
    plt.savefig(filePathToSavePlot, dpi=100)
    if showPlots:
        plt.show()

    return

def evaluate(artefacts, selectors, test, evaluator):
    results = []

    for s in selectors:
        selected_artefacts = s.select(artefacts)

        evaluation_results = []
        testing_combinations = test.create_combinations(selected_artefacts)
        for testing_combination in testing_combinations:
            single_evaluation = evaluator.evaluate(testing_combination)
            evaluation_results.append(single_evaluation)

        selectors_result = evaluator.combine_evaluations(evaluation_results)
        selectors_result[0].set_selector(s)
        results.append(selectors_result[0])
        log(selectors_result[0])
        if len(results) % 500 == 0: print_best(results)
        
        #### TEMP######
        actual = selectors_result[2]
        pred = selectors_result[1]
        from sklearn.metrics import confusion_matrix
        labels = ['MSA', 'PD', 'PSP', 'CTRL']
        confMat = confusion_matrix(actual, pred, labels=labels)
        sumaPoRedovima = confMat.astype('float').sum(axis=1)
        confMatPerc = [gore/dole for gore, dole in zip(confMat, sumaPoRedovima)]
        confMatPerc = np.matrix(confMatPerc)*100   
        filePathToSavePlot = r'D:\GIT\FingerTapping\figures\cm.jpg'
        plotConfMat(confMat, confMatPerc, filePathToSavePlot, 1)

    return results


def print_best(results):
    print_best_n(results)
    print_best_one(results)


def print_best_one(results):
    best_val = 0
    best_selectors = []
    last_in = None
    last_matrix = None
    for result in results:
        for key in result.results:
            val = result.results[key][0]
            if val == best_val:
                if last_in != result:
                    best_selectors.append(result)
                    last_in = result
            elif val > best_val:
                best_val = val
                best_selectors.clear()
                best_selectors.append(result)
                last_in = result
                last_matrix = result.results[key][1]
            else:
                continue

    print('---------Best results--------')
    print(last_matrix)
    log(best_selectors, './results/resultML')

    return


def print_best_n(results, n=10):
    results.sort(key=sorting_index, reverse=True)
    print('---------Best N={} results--------'.format(n))
    best_selectors = results[0:n]
    best_selectors.reverse()
    log(best_selectors, './results/n_resultML')
    return


def sorting_index(result):
    best_val = 0
    for key in result.results:
        val = result.results[key][0]
        if val > best_val:
            best_val = val
    selector = result.selector.selector
    cnt = 0
    while selector != 0:
        if selector & 1 == 1: cnt = cnt + 1
        selector = selector // 2

    return [best_val, cnt]


def log(result, file_name='./results/logML'):
    print(result)

    suffix, first = get_time_suffix()
    file_name = file_name + "_" + suffix + ".txt"

    with open(file_name, 'a') as file:
        if first: file.write(ArtefactFilter.get_filtered_names())
        file.write(str(result))
        file.close()

    return


def get_time_suffix():
    first = False
    if Parameters.file_suffix is None:
        Parameters.file_suffix = datetime.now()
        Parameters.file_suffix = Parameters.file_suffix.strftime("%Y-%m-%d_%H-%M-%S")
        first = True
    return Parameters.file_suffix, first
