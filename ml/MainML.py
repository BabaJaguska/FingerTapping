import Parameters
import Signal
import ArtefactExtractor, ArtefactSelectorGenerator, \
    ArtefactEvaluatorGenerator, ArtefactEvaluator, \
    ArtefactTestGenerator, ArtefactFilter, ArtefactNormalisator

import numpy as np
import itertools
from matplotlib import pyplot as plt
import os
    
def plotConfMat(cm, cmPerc, filePathToSavePlot='CM.tiff'):
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
    
    small = 35
    medium = 37
    large = 40
    huge = 45

    plt.figure(figsize=(28, 22))
    plt.imshow(cmPerc)
    cbar = plt.colorbar()
    N_classes = len(cm)
    classes = ['HC', 'MSA', 'PD', 'PSP']
 
    for i, j in itertools.product(range(N_classes), range(N_classes)):
        plt.text(i, j, '{}%\n(N={})'.format(round(cmPerc[j, i], 2), cm[j, i], 'd'),
                 horizontalalignment="center",
                 color='white' if cmPerc[j, i] < 40 else "black",
                 size=small)
    tick_marks = np.arange(N_classes)

    plt.xticks(tick_marks, classes, rotation=45, size=medium)
    plt.yticks(tick_marks, classes, size=medium)
    plt.ylabel('Predicted label', size=huge)
    plt.xlabel('\nTrue label', size=huge)

    plt.title('Confusion matrix\n', size=huge)
    cbar.ax.tick_params(labelsize=25) 
    fig1 = plt.gcf()
    plt.savefig(filePathToSavePlot, dpi=100)

    plt.show()
    
    return   




# def main():
measurements = Signal.load_all(Parameters.default_root_path)

#%%

# measurements = []
# ids = []

# for measurement in measurementsAll:
#     personID = measurement.id.split('.')[:2]
#     if personID in ids:
#         continue
    
#     if measurement.tap_task[:2] == 'RH':
#         ids.append(personID)
#         measurements.append(measurement)


#%%


Signal.show_signals_info(measurements, plot=Parameters.show_all)

artefacts = ArtefactExtractor.extract(measurements)
print(len(artefacts))

# artefacts = ArtefactNormalisator.normalise(artefacts) # !!!!
print(len(artefacts))

artefacts = ArtefactFilter.filtering(artefacts)
print(ArtefactFilter.get_filtered_names())
feature_names = ArtefactFilter.get_filtered_names()
feature_names = feature_names.split('\n')

selectors = ArtefactSelectorGenerator.select(artefacts, selection_type='SEQUENCE',
                                             start_index=63, end_index=64,
                                             initial_selectors=[0b111111])

print(selectors)

test = ArtefactTestGenerator.generate()
print(test)

evaluator = ArtefactEvaluatorGenerator.get_evaluator()
print(evaluator)

results, importances = ArtefactEvaluator.evaluate(artefacts, selectors, test, evaluator)
ArtefactEvaluator.print_best(results)

    # return


# main()


#%% features

featsCTRL = np.array([a.values for a in artefacts if a.result == 'CTRL'])
featsMSA = np.array([a.values for a in artefacts if a.result == 'MSA'])
featsPD = np.array([a.values for a in artefacts if a.result == 'PD'])
featsPSP = np.array([a.values for a in artefacts if a.result == 'PSP'])

meanz = []
meanz.append(np.mean(featsCTRL, axis=0))
meanz.append(np.mean(featsMSA, axis=0))
meanz.append(np.mean(featsPD, axis = 0))
meanz.append(np.mean(featsPSP, axis = 0))
meanz = np.array(meanz)

feature_order = [1, 4, 0, 3, 2, 5 ]
ff = np.array(feature_names)
ff = ff[feature_order]

meanz = meanz[:,feature_order]
print(feature_names)
print(meanz)

##

meanz = []
meanz.append(np.quantile(featsCTRL,0.25 , axis=0))
meanz.append(np.quantile(featsMSA,0.25 , axis=0))
meanz.append(np.quantile(featsPD,0.25 , axis = 0))
meanz.append(np.quantile(featsPSP,0.25 , axis = 0))
meanz = np.array(meanz)

feature_order = [1, 4, 0, 3, 2, 5 ]
ff = np.array(feature_names)
ff = ff[feature_order]

meanz = meanz[:,feature_order]
print(feature_names)
print(meanz)

##

meanz = []
meanz.append(np.quantile(featsCTRL,0.75 , axis=0))
meanz.append(np.quantile(featsMSA,0.75 , axis=0))
meanz.append(np.quantile(featsPD,0.75 , axis = 0))
meanz.append(np.quantile(featsPSP,0.75 , axis = 0))
meanz = np.array(meanz)

feature_order = [1, 4, 0, 3, 2, 5 ]
ff = np.array(feature_names)
ff = ff[feature_order]

meanz = meanz[:,feature_order]
print(feature_names)
print(meanz)


#%% plot

ctrl = 18
msa =  119 #159 #203
pd = 293
psp = 435 #370

small = 20
smaller = 18
large = 28
mid = 24

nsec = 4*200
t = np.linspace(0,4,nsec)
plt.cm.gray
plt.figure(figsize = (20,20))




plt.subplot(421)
plt.plot(t,measurements[msa].gyro1x[:nsec], linewidth=4, color='0.6')
plt.plot(t,measurements[msa].gyro1y[:nsec], linestyle='dashed', color='0.4', linewidth=2)
plt.plot(t,measurements[msa].gyro1z[:nsec], color='0.1',linewidth=1)
plt.title('MSA participant -  THUMB', size=large)
plt.xlabel('t [s]', size = small)
plt.ylabel(r'$\omega [rad/s]$', size = mid)
plt.xticks(size=smaller)
plt.yticks(size = small)
plt.ylim([-10,16])

plt.subplot(422)
plt.plot(t,measurements[msa].gyro2x[:nsec], linewidth=4, color='0.5')
plt.plot(t,measurements[msa].gyro2y[:nsec], linestyle='dashed', color='0.4', linewidth=2)
plt.plot(t,measurements[msa].gyro2z[:nsec], color='0.1', linewidth=1)
plt.title('MSA participant -  INDEX', size=large)
plt.xlabel('t [s]', size = small)
plt.ylabel(r'$\omega [rad/s]$', size = mid)
plt.xticks(size=smaller)
plt.yticks(size = small)
plt.ylim([-21,31])

plt.subplot(423)
plt.plot(t,measurements[pd].gyro1x[:nsec], linewidth=4, color='0.5')
plt.plot(t,measurements[pd].gyro1y[:nsec], linestyle='dashed', color='0.4', linewidth=2)
plt.plot(t,measurements[pd].gyro1z[:nsec], color='0.1', linewidth=1)
plt.title('PD participant -  THUMB', size=large)
plt.xlabel('t [s]', size = small)
plt.ylabel(r'$\omega [rad/s]$', size = mid)
plt.xticks(size=smaller)
plt.yticks(size = small)
plt.ylim([-10,16])

plt.subplot(424)
plt.plot(t,measurements[pd].gyro2x[:nsec], linewidth=4, color='0.5')
plt.plot(t,measurements[pd].gyro2y[:nsec], linestyle='dashed', color='0.4', linewidth=2)
plt.plot(t,measurements[pd].gyro2z[:nsec], color='0.1', linewidth=1)
plt.title('PD participant -  INDEX', size=large)
plt.xlabel('t [s]', size = small)
plt.ylabel(r'$\omega [rad/s]$', size = mid)
plt.xticks(size=smaller)
plt.yticks(size = small)
plt.ylim([-21,31])


plt.subplot(425)
plt.plot(t,measurements[psp].gyro1x[:nsec], linewidth=4, color='0.5')
plt.plot(t,measurements[psp].gyro1y[:nsec], linestyle='dashed', color='0.4', linewidth=2)
plt.plot(t,measurements[psp].gyro1z[:nsec], color='0.1', linewidth=1)
plt.title('PSP participant -  THUMB', size=large)
plt.xlabel('t [s]', size = small)
plt.ylabel(r'$\omega [rad/s]$', size = mid)
plt.xticks(size=smaller)
plt.yticks(size = small)
plt.ylim([-10,16])

plt.subplot(426)
plt.plot(t,measurements[psp].gyro2x[:nsec], linewidth=4, color='0.5')
plt.plot(t,measurements[psp].gyro2y[:nsec], linestyle='dashed', color='0.4', linewidth=2)
plt.plot(t,measurements[psp].gyro2z[:nsec], color='0.1', linewidth=1)
plt.title('PSP participant -  INDEX', size=large)
plt.xlabel('t [s]', size = small)
plt.ylabel(r'$\omega [rad/s]$', size = mid)
plt.xticks(size=smaller)
plt.yticks(size = small)
plt.ylim([-21,31])

plt.subplot(427)
plt.plot(t,measurements[ctrl].gyro1x[:nsec], linewidth=4, color='0.5')
plt.plot(t,measurements[ctrl].gyro1y[:nsec], linestyle='dashed', color='0.4', linewidth=2)
plt.plot(t,measurements[ctrl].gyro1z[:nsec], color='0.1',linewidth=1)
plt.title('HC participant -  THUMB', size=large)
plt.xlabel('t [s]', size = small)
plt.ylabel(r'$\omega [rad/s]$', size = mid)
plt.xticks(size=smaller)
plt.yticks(size = small)
plt.ylim([-10,16])


plt.subplot(428)
plt.plot(t,measurements[ctrl].gyro2x[:nsec], linewidth=4, color='0.5')
plt.plot(t,measurements[ctrl].gyro2y[:nsec], linestyle='dashed', color='0.4', linewidth=2)
plt.plot(t,measurements[ctrl].gyro2z[:nsec], color='0.1', linewidth=1)
plt.title('HC participant -  INDEX', size=large)
plt.xlabel('t [s]', size = small)
plt.ylabel(r'$\omega [rad/s]$', size = mid)
plt.legend(['X', 'Y', 'Z'],  prop={'size': 20})
plt.xticks(size=smaller)
plt.yticks(size = small)
plt.ylim([-21,31])

plt.tight_layout()

if not os.path.exists('./figs'):
    os.mkdir('./figs')
plt.savefig('./figs/fig2brt.jpg',
            dpi=100)
plt.show()



