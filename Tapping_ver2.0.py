# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:37:11 2019

@author: minja
"""
#%%
# =============================================================================
#  load stuff
# =============================================================================
import numpy as np
import time

from tqdm import tqdm
#import pandas as pd

from sklearn.model_selection import StratifiedKFold
#import keras
from keras.models import Model, load_model
from keras.layers import Input, Conv1D, Flatten, Dropout, MaxPooling1D, Dense, Reshape,UpSampling1D
from keras.layers import Activation, BatchNormalization, concatenate
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.constraints import max_norm
import os
# from numpy import ma

from sklearn.metrics import confusion_matrix
import itertools
# from keras import backend as K   


from readAndEncode import *

#%% PARAMETERS

root = r'C:\data\icef\tapping\raw data'
path = r'C:\data\icef\tapping'

nSeconds = 4
rightHandOnly = 1
autoencoderInitiated = 0
nFolds = 3
ConvLayers = [2]   
epochs = 100
batch_sizes = [16]
kernelSizes = [9] #so far for pos in best 11
res = []
kernel_constraints = [2]
nDenseUnits = [16]
nInitialFilters = [8]

##
nNUnits = [32]
dropoutRate = [0.3]

##

plt.rcParams['image.cmap'] = 'magma'
plt.rcParams['font.family'] = 'serif'

 

#%% FUNCTION DEFINITIONS






# =============================================================================
# THE MODEL
# =============================================================================

def defCallbacks(weightFile):
    
    checkpoint = ModelCheckpoint(weightFile,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only = True,
                                 save_weights_only = True,
                                 mode='max')
    early = EarlyStopping(monitor='val_loss',
                          patience = 20,
                          verbose = 1,
                          mode='min')
#     def step_decay(epoch):
#         initial_lrate = 0.001
#         rate_drop = 0.25
#         nEpochs = 5
#         lrate = initial_lrate * math.pow(rate_drop, math.floor(epoch/nEpochs))
#         return lrate 
    
#     lrate = LearningRateScheduler(step_decay, verbose = 1)
    lr_plateau = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2,
                                  patience = 8, min_lr = 0.0000000001,
                                  verbose = 1)
    
    return [checkpoint, early, lr_plateau]

def examineHistory(history,modelName):
    

    # plot Accuracy over Epochs
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train Acc','Val Acc'])
    plt.title('Accuracy for {} over epochs'.format(modelName))
    plt.show()


    # plot Loss over Epochs
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train Loss','Val Loss'])
    plt.title('Loss for {} over epochs'.format(modelName))
    plt.show()
    
    print("Max val accuracy: ", max(history['val_accuracy']))
    print("Max train accuracy: ", max(history['accuracy']))
        
    return max(history['val_accuracy']),max(history['accuracy'])



def CNNModel(inputShape, nConvLayers, kernel_size, kernel_constraint,nUnits,initialFilters):
    
    input1 = Input(shape = inputShape)

    #convolutions
    x = input1
    
    
    for i in range(0,nConvLayers):
        
        nFilters = 128 if initialFilters*(2**(i))>128 else initialFilters*(2**(i))
        #inside = 3 if i>1 else 2
        inside = 2
        k = 11 if i ==0 else kernel_size
        for temp in range(0,inside):
            x = Conv1D(filters = nFilters,
                  kernel_size = k,
                  padding = 'same',
                  strides = 2,
                  kernel_initializer = 'he_normal',
                  kernel_constraint = max_norm(kernel_constraint),
                  name = 'Conv1x5{}{}'.format(i,temp))(x)
            
            x = Activation('relu',name='ReLu{}{}'.format(i,temp))(x)
            x = BatchNormalization()(x)
            
            x = MaxPooling1D((2),
                      padding = 'same')(x)
    
    x = Dropout(0.3)(x) #0.6!
    
    # Fully connected
    x = Flatten()(x)
    
    x = Dense(nUnits,
              kernel_constraint = max_norm(1),
              kernel_initializer = 'he_normal')(x)
    x = Activation('relu', name = 'reLU_dense')(x)
    x = Dropout(0.3)(x)
    
    x = Dense(4)(x)
    x = Activation('softmax',name = 'Softmax')(x)

    m = Model(input1,x)
    return m



def fitModel(model, modelName, Xtrain, Ytrain, Xval, Yval, epochs,batch_size):
    
    tic = time.time()
    
    # make file name
    tm = time.gmtime()
    weightFile = path+'shuffleBEST_WEIGHTS{}.{}.{}.{}.{}.{}.h5'.format(modelName,tm[2],tm[1],tm[0],tm[3]+1,tm[4])
    
    #define callbacks
    callbacks = defCallbacks(weightFile)
    
    # FIT THE MODEL
    history = model.fit(x = Xtrain, y = Ytrain,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data = (Xval,Yval),
                        callbacks = callbacks,
                        shuffle=True)
    toc = time.time()
    print("Finished training in {} min ({} h)".format(round((toc-tic)/60,2),round((toc-tic)/3600,2)))

    
    # Save the weights
    #model.save_weights(str(modelName)+'.h5') # ???????
    
    return history


# def examineLayerOutputs():
    #"pooling" in layername or\
    #"ReLu" in layername or\
    #"reLU" in layername or\
    #"Softmax" in layername or\
        

    # inp = model.input
    # outs = [layer.output for layer in model.layers if hasWeights(layer.name)]
    # functor = K.function([inp]+ [K.learning_phase()],outs)
    # izlazi = functor([XtrainRight,0])



def CNNModelAuto(inputShape, nConvLayers, kernel_size, kernel_constraint,initFilters,nUnits):
    
    input1 = Input(shape = inputShape)

    #convolutions
    x = input1
    
    
    for i in range(0,nConvLayers):
        
        nFilters = 128 if initFilters*2**(i)>128 else initFilters*2**(i)
        #inside = 3 if i>1 else 2
        inside = 2
        for temp in range(0,inside):
            x = Conv1D(filters = nFilters,
                  kernel_size = kernel_size,
                  padding = 'same',
                  strides = 1,
                  kernel_initializer = 'he_normal',
                  kernel_constraint = max_norm(kernel_constraint),
                  name = 'Conv1x5{}{}'.format(i,temp))(x)
            
            x = Activation('relu',name='ReLu{}{}'.format(i,temp))(x)
            x = BatchNormalization()(x)
            
            x = MaxPooling1D((2),
                      padding = 'same')(x)
    
    x = Dropout(0.3)(x)
    
    # Fully connected
    x = Flatten(name ='flatWhite')(x)
    
#    x = Dense(nUnits,
#              kernel_constraint = max_norm(kernel_constraint),
#              kernel_initializer = 'he_normal')(x)
#    x = Activation('relu', name = 'reLU_dense')(x)
#    x = Dropout(0.5,name='FCstuffEncoded')(x)
    
    
    filtTemp = 128 if nConvLayers>2 else (nConvLayers)*initFilters
    rShape = (int(np.ceil(inputShape[0]/(2**(2*nConvLayers)))),filtTemp)
#    print(rShape)
#    x = Dense(int(np.ceil(inputShape[0]/(2**(2*nConvLayers))))*filtTemp,
#              activation='relu',
#              kernel_initializer='he_normal',
#              name='denseDude')(x)
    x = Reshape(rShape)(x)
    # decoder
    
    for i in range(0,nConvLayers):
        
        nFilters = 128 if i <nConvLayers-2 else initFilters*(nConvLayers-i)
        
        for temp in range(0,2):
            x = Conv1D(filters = nFilters,
                  kernel_size = kernel_size,
                  padding = 'same',
                  strides = 1,
                  kernel_initializer = 'he_normal',
                  kernel_constraint = max_norm(kernel_constraint))(x)
            
            x = Activation('relu')(x)
            x = BatchNormalization()(x)
            
            x = UpSampling1D(2)(x)

    x = Conv1D(kernel_size = 1, strides = 1, filters = 6,
               padding = 'same',
               kernel_initializer = 'he_normal',
               kernel_constraint = max_norm(kernel_constraint))(x)
    x = Activation('sigmoid')(x)
            
    m = Model(input1,x)
    #features = m.get_layer('flatWhite').output
    features = m.get_layer('flatWhite').output
    encoder = Model(input1,features)
 
    return m,encoder



#utilize initialization
def modelFromAuto(nUnits,dropoutRate):
    path='C:/data/tapping/'
    encoder = load_model(path+'encoderCEOshuffleRIGHT.h5')
    x = encoder.output
    x = Dense(nUnits,kernel_constraint = max_norm(1),
              kernel_initializer = 'he_normal',
              activation = 'relu')(x)
    x = Dropout(dropoutRate)(x)
    x = Dense(4, activation = 'softmax')(x)
    
    m = Model(encoder.input,x)
    return m

def predictSignal(signalX, signalY, model, verbose = 1):
    signalX = np.expand_dims(signalX,axis=0)
    prediction = model.predict(signalX)
    prediction = prediction[0]
    prediction = np.round(np.float32(prediction),2)
    
    d = {0: 'HC',
        1: 'MSA',
        2: 'PD',
        3: 'PSP'}
    
    

    actualDiagnosis = d[np.argmax(signalY)]
    predictedDiagnosis = d[np.argmax(prediction)]
    
    if verbose:
        print('predictedDiagnosis: ',predictedDiagnosis)
        print('actualDiagnosis: ', actualDiagnosis)
        print('Certainty: \nHC: {} \nMSA: {} \nPD: {} \nPSP: {}'.format(prediction[0],
                                                                      prediction[1],
                                                                      prediction[2],
                                                                      prediction[3]))
        print('#################################################')
   
    
    return {"Predicted": predictedDiagnosis, "Actual": actualDiagnosis}

# =============================================================================
# METRICS
# =============================================================================


def confuseMat(Xtest, Ytest,model):
    p, a = [], []
    for X,Y in zip(Xtest,Ytest):
        temp = predictSignal(X,Y,model,verbose=0)
        p.append(temp['Predicted'])
        a.append(temp['Actual'])
    confMat = confusion_matrix(a,p)
    print(confMat)
    sumaPoRedovima = confMat.astype('float').sum(axis=1)
    confMatPerc = [gore/dole for gore,dole in zip(confMat,sumaPoRedovima)]
    confMatPerc = np.matrix(confMatPerc)*100
    
    return [confMat,confMatPerc]


def plotConfMat(cm,cmPerc):
    s = 19
    sTitle = 22
    plt.figure(figsize = (10,10))
    plt.imshow(cmPerc)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=s)
    for i,j in itertools.product(range(4),range(4)):
        plt.text(i,j,'{}%\n({})'.format(round(cmPerc[j,i],2),cm[j,i],'d'),
                horizontalalignment = "center",
                color = "white" if cmPerc[j,i]<60 else "black",
                size = s)
    tick_marks = np.arange(4)
    classes = ['HC','MSA','PD','PSP']
    plt.xticks(tick_marks,classes,rotation = 45,size = s)
    plt.yticks(tick_marks,classes,size = s)
    plt.ylabel('True label',size = sTitle)
    plt.xlabel('\nPredicted label',size = sTitle)
    #plt.style.use(['tableau-colorblind10'])
    #plt.rcParams['image.cmap'] = 'viridis'
    plt.title('Confusion matrix - Test B \n', size = sTitle)
    plt.show()
    return



#precision and recall

def metrics(confmat):
    allOfClass = np.sum(confmat,axis=1)
    allAsClass = np.sum(confmat,axis=0)
    correctly = np.diag(confmat)
    precision = np.round(np.divide(correctly, allAsClass)*100,2)
    recall = np.round(np.divide(correctly, allOfClass)*100,2)
    allall = np.sum(confmat)
    allcorrect = np.sum(correctly)
    acc = np.round(allcorrect*100/allall,2)
    return acc, precision, recall


    
def hasWeights(layername):
    ok = True
    if "activation" in layername or\
    "dropout" in layername or\
    "FLAT" in layername  or\
    "flatten" in layername or\
    "input" in layername or\
    "reshape" in layername:
        ok = False
    return ok

#%%
# =============================================================================
#     READ THE DATA
# =============================================================================


allPatientFolders, allPatientDiagnoses = getUniquePatients(root)




# ====== SPLIT DIRPATHS INTO FOLDS =======



skf = StratifiedKFold(nFolds, shuffle = True, random_state = 123)
k = 1

for train_index, test_index in skf.split(allPatientFolders, allPatientDiagnoses):
    
    print("Processing fold {} of {}".format( k, nFolds))
    k +=1
    print("TRAIN:", train_index, "\nTEST:", test_index)
    print('==================================================')
    
    
    trainPatientFolders = allPatientFolders[train_index]
    trainPatientDiagnoses = allPatientDiagnoses[train_index]
    
    testPatientFolders = allPatientFolders[test_index]
    testPatientDiagnoses = allPatientDiagnoses[test_index]


    
    trainSigs = []
    testSigs = []
    
    for patientFolder in tqdm(trainPatientFolders):
        for r,_,files in os.walk(patientFolder):
            for file in files:
                trainSigs.append(readAdjust(os.path.join(r,file)))
                
                
    for patientFolder in tqdm(testPatientFolders):
        for r,_,files in os.walk(patientFolder):
            for file in files:
                testSigs.append(readAdjust(os.path.join(r,file)))

        
        
    
    # =============================================================================
    #     ckeck if stuff is okay
    # =============================================================================
    
    # pick an example signal
    exampleSig = trainSigs[8]
    
    temp = [s.sumUp()['MATCHING_LENGTHS'] for s in trainSigs + testSigs]
    if len(set(temp))==1:
        print('All signals contain gyro and fsr data of the same length')
    else:
        print('Some files contain data of unequal lengths')
    
        
    
    exampleSig.plotSignals([0,10])
    print('Example signals plotted for {}'.format(exampleSig.id))
    print(exampleSig.sumUp())



# Minimal length of signals?

    m = 10000
    for ix,sig in enumerate(trainSigs + testSigs):
        if sig.tappingLength < m:
            m = sig.tappingLength
            min_id = sig.id
            
    # The shortest signal is sigs[86], corresponding to a CTRL, rhec. Lasts a total of 12s, and 7.25s of active tapping.
    # Going to remove those with active signal <10s...maybe
    print('Shortest length: {}s, found for signal id: {} '.format(m, min_id))



# Check data distribution
    classDistribution(trainSigs)
    classDistribution(testSigs)






# =============================================================================


    if rightHandOnly:
        
        print('\n>>>>Using right hand recordings only<<<<\n')
        testSigsRight = [x  for x in testSigs if x.isRightHand()]
        # testSigsLeft = [x  for x in testSigs if (x.tap_task=='LHEO' or x.tap_task=='LHEC')]
        
        # dataValRight = [x  for x in dataVal if (x.tap_task=='RHEO' or x.tap_task=='RHEC')]
        # dataValLeft = [x  for x in dataVal if (x.tap_task=='LHEO' or x.tap_task=='LHEC')]
        
        trainSigsRight = [x  for x in trainSigs if x.isRightHand()]
        # trainSigsLeft = [x  for x in trainSigs if (x.tap_task=='LHEO' or x.tap_task=='LHEC')]
        
        Xtrain, Ytrain = cropAndReshape(trainSigsRight,nSeconds)
        Xtest, Ytest = cropAndReshape(testSigsRight,nSeconds)
    else:
        Xtrain, Ytrain = cropAndReshape(trainSigs,nSeconds)
        Xtest, Ytest = cropAndReshape(testSigs,nSeconds)
    
    
    # if autoencoderInitiated:
    #     trainSigsAutoencoder = np.concatenate((trainSigs,testSigsLeft),0)
    #     trainSigsAutoencoder.shape
    #     XtrainA, YtrainA = cropAndReshape(trainSigsAutoencoder,nSeconds)
    #     print('>>>>Autoencoder initialization<<<<')
    # else:

    #     print('>>>>Random initialization<<<<')
    
    
#XtestMulti, YtestMulti = cropAndReshape(testSigs,nSeconds)
# XvalA, YvalA = cropAndReshape(dataVal,nSeconds)


# XvalRight, YvalRight = cropAndReshape(dataValRight,nSeconds)

#==============================================================================
# =============================================================================
#     >>>>>>>>>>>>>>>>>> CLASSIFIER <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# =============================================================================
# =============================================================================
# THE DATA TO USE
# =============================================================================



#XtestMulti, YtestMulti = cropAndReshape(testSigs,nSeconds)
# Xval, Yval = cropAndReshape(dataVal,nSeconds)



# =============================================================================
#     TRAAAAAIIIIIINNNNNN!! (NO AUTOENCODER. RANDOM INITIALIZATION)
# =============================================================================

#modelDepths = []

    for kernel_size, nConvLayers, kernel_constraint, nunits, initialFilters, batch_size  in \
    zip(kernelSizes, ConvLayers, kernel_constraints, nNUnits, nInitialFilters, batch_sizes):
        


        vals = []
        trains = []
        testa = []

        
        model = CNNModel((Xtrain.shape[1],Xtrain.shape[2]),
                         nConvLayers,
                         kernel_size,
                         kernel_constraint,
                         nunits,
                         initialFilters)
        model.summary()
        #opt = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
        optA = optimizers.Adam(lr=0.001)
        model.compile(optimizer = optA,
                 loss='categorical_crossentropy',
                 metrics = ['accuracy'])
        #model.summary()

        modelName = 'CNNShuffled'+str(batch_size)+'Batch'+ \
        str(nConvLayers)+\
        'ConvLayers'+\
        str(kernel_size)+\
        'KERNEL'+str(nunits)+\
        'DenseUnits'+str(initialFilters)+\
        'initFilt'
        #saveModelTopology(model,modelName)
        model.save(path+modelName+'CEO.h5')

        print('TRAINING...')

        history = fitModel(model,
                           modelName,
                           Xtrain, Ytrain,
                           Xtest, Ytest, 
                           epochs,
                           batch_size)
        valAcc,trainAcc = examineHistory(history.history,modelName)
        vals.append(valAcc)
        trains.append(trainAcc)
        actual=model.evaluate(Xtest,Ytest)
        testa.append(actual[1])
        
        
        cm, cmPerc = confuseMat(Xtest,Ytest,model)
        plotConfMat(cm,cmPerc)
            
            
        res.append({'history':history,
                    'kernelSize':kernel_size,
                    'nConvLayers':nConvLayers,
                    'batchSize':batch_size,
                    'constraint':kernel_constraint,
                    'nDenseUnits':nunits,
                    'nInitialFilters':initialFilters,
                    'bestValAcc':vals,
                    'bestTrainAcc':trains,
                    'meanBestVal': np.mean(vals),
                    'meanBestTrain':np.mean(trains),
                    'confMats': cm,
                    'testAccuracy:':testa})
                
        
        
res       


print('AAAAAAAAAAAAAAAAAAAAAA')
#%% AUTOENCODER INITIATED

if autoencoderInitiated:
    inputShape = (XtrainRight.shape[1],XtrainRight.shape[2])
    [autoencoder,encoder] = CNNModelAuto(inputShape,3,11,3,32,64)
    autoencoder.summary()
    autoencoder.compile(optimizer = 'adam',
                        metrics = ['accuracy'],
                        loss = 'binary_crossentropy')
    
    
    
    
    
    
    callbacks = defCallbacksA(path+'autoencoderWeightsRIGHTshuffle.h5')
    historyA = autoencoder.fit(XtrainRight,XtrainRight,
                              epochs = 100,
                              batch_size = 64,
                              validation_data = (XvalRight,XvalRight), #XvalA,XvalA
                              callbacks = callbacks,
                              shuffle = True)
    
    
    
    autoencoder.load_weights(path + 'autoencoderWeightsRIGHTshuffle.h5')
    for l1,l2 in zip(encoder.layers,autoencoder.layers[:27]):
        l1.set_weights(l2.get_weights())
    
    
    encoder.save(path+'encoderCEOshuffleRIGHT.h5')
    
    
    examineHistory(historyA.history,'a')





    
   # predict some vals
    reconstructed = autoencoder.predict(XvalRight[22:24,:,:])
    t = np.arange(0,len(XvalRight[22])/200,1/200)
    plt.figure(figsize = (10,5))
    plt.plot(t,XvalRight[22])
    plt.xlabel('t[s]')
    plt.title('Original normalized 6-channel gyroscope signal')
    plt.show()
    plt.figure(figsize = (10,5))
    plt.plot(t,reconstructed[0])
    plt.title('Autoencoder reconstructed signal')
    plt.xlabel('t[s]')
    plt.show()


    #blabalbal
     
    for rate in  dropoutRate:
        for units in nUnits:
            vacc = []
            tacc = []
            a = []
            for i in range(5):
                
                mA = modelFromAuto(units,rate)
    #            for l in mA.layers[:27]:
    #                l.trainable = False
                op = optimizers.Adam(lr = 0.001)
                #op1 = optimizers.SGD(lr = 0.0001,momentum = 0.8, nesterov= True )
                mA.compile(optimizer = op,
                           metrics = ['accuracy'],
                           loss = 'categorical_crossentropy')
                tm = time.gmtime()
                callbacks = defCallbacks(path+'autoencoderBasedRightShuffled{}-{}-{}.{}.{}.{}.h5'.format(tm[0],tm[1],tm[2],tm[3],tm[4],tm[5]))
                h = mA.fit(XtrainRight,YtrainRight,
                           validation_data = (XvalRight,YvalRight),
                           epochs = 150,
                           batch_size = 16,
                           callbacks = callbacks)
                
                vA,tA = examineHistory(h.history,'bla')
                actual=mA.evaluate(XtestRight,YtestRight)
                vacc.append(vA)
                tacc.append(tA)
                a.append(actual)
            res.append({'rate':rate,
                        'nUnits':units,
                        'vallAcc':vacc,
                        'trainAcc':tacc,
                        'meanval':np.mean(vacc),
                        'testAcc':a[1]})
           
    res




    #svaki pojedinacno
    import random
    accEach = []
    predEach = []
    predAll=[]
    dijagnoze=[]
    najmanjeCropova = 100
    srednjePred=[]
    predProb = []
    for datic in testSigsRight:
        tempX,tempY = cropAndReshape([datic],nSeconds)
        print(tempX.shape)
        tempCropova = tempX.shape[0]
        if tempCropova<najmanjeCropova:
            najmanjeCropova = tempCropova
            ##
        random.seed(123)
        idx = range(4)
        _,tempAcc = mA.evaluate(tempX[idx],tempY[idx])
        # a vidi sad koje tacno brojeve vraca, ne samo max index
        predikcije = np.mean(mA.predict(tempX[idx]),axis=0)
        paMean = np.argmax(predikcije)
        srednjePred.append(paMean)
        predProb.append(predikcije)
        tempPred = [np.argmax(p) for p in mA.predict(tempX[idx])]
        pa=[np.argmax(p) for p in mA.predict(tempX[idx])]
        predAll.append(pa)
        accEach.append(tempAcc) # na onoliko koliko si izabrala odbiraka
        predEach.append(tempPred)
        dijagnoze.append(datic.diagnosis)
    
        ##
    
    print('najmanji broj kropova u fajlu je: ',najmanjeCropova)
    nule = 0
    jediniceBas = 0
    for a in accEach:
        if a<=0.5:
            nule+=1
        if a>0.99999:
            jediniceBas+=1
            
            
    print('zapravo accuracy: ',100*(1-nule/len(testSigsRight)))
    
    #    
    
    kodiraniSmoothies = [kodiraj(pred) for pred in srednjePred]
    cmm = confusion_matrix(dijagnoze,kodiraniSmoothies)
    cmmPerc = [c/s for c,s in zip(cmm,np.sum(cmm,axis=1))]
    cmmPerc=np.matrix(cmmPerc)*100
    plotConfMat(cmm,cmmPerc)
    
    #%
    from keras.utils import plot_model
    
    plot_model(mA,path+'juhuModelSlika.png',show_shapes=True)
    
    
    
    
    #%
    
    temp = izlazi[3] #maxpool 1
    #plt.figure(figsize = (12,8))
    #plt.plot(temp[12,:,0]) #osoba red br 12, kanal 0ti
    #plt.show()
    #plt.plot(temp[12,:,1]) #osoba red br 12, kanal 1vi
    #plt.show()
    #plt.plot(temp[12,:,10]) #osoba red br 12, kanal 10i
    #plt.show()
    
    plt.plot(temp[98,:,:]) #osoba red br 12, svih 32 kanala  #MSA
    plt.title("MSA, CONV1")
    plt.show()
    
    plt.plot(temp[555,:,:]) #osoba red br 122, svih 32 kanala #CTRL
    plt.title("CTRL, CONV1")
    plt.show()
    
    
    plt.plot(temp[66,:,:]) #osoba red br 122, svih 32 kanala #PD
    plt.title("PD, CONV1")
    plt.show()
    
    plt.plot(temp[1100,:,:]) #osoba red br 122, svih 32 kanala #PSP
    plt.title("PSP CONV1")
    plt.title("PSP, CONV1")
    plt.show()
    ###
    
    temp = izlazi[7] #maxpool posle 2 conv
    plt.plot(temp[98,:,:]) #osoba red br 12, svih 32 kanala  #MSA
    plt.title("MSA, CONV2")
    plt.show()
    
    plt.plot(temp[555,:,:]) #osoba red br 122, svih 32 kanala #CTRL
    plt.title("CTRL CONV2")
    plt.show()
    
    plt.plot(temp[66,:,:]) #osoba red br 122, svih 32 kanala #PD
    plt.title("PD, CONV2")
    plt.show()
    
    plt.plot(temp[1100,:,:]) #osoba red br 122, svih 32 kanala #PSP
    plt.title("PSP CONV2")
    plt.show()
    ###
    temp = izlazi[11] #max posle conv3
    plt.plot(temp[98,:,:]) #osoba red br 12, svih 32 kanala  #MSA
    plt.title("MSA, CONV3")
    plt.show()
    
    plt.plot(temp[555,:,:]) #osoba red br 122, svih 32 kanala #CTRL
    plt.title("CTRL CONV3")
    plt.show()
    
    plt.plot(temp[66,:,:]) #osoba red br 122, svih 32 kanala #PD
    plt.show()
    
    plt.plot(temp[1100,:,:]) #osoba red br 122, svih 32 kanala #PSP
    plt.title("PSP CONV3")
    plt.show()
    
    ##
    temp = izlazi[15] #max posle conv 4
    plt.plot(temp[98,:,:]) #osoba red br 12, svih 32 kanala  #MSA
    plt.title("MSA, CONV4")
    plt.show()
    
    plt.plot(temp[555,:,:]) #osoba red br 122, svih 32 kanala #CTRL
    plt.title("CTRL CONV4")
    plt.show()
    
    plt.plot(temp[66,:,:]) #osoba red br 122, svih 32 kanala #PD
    plt.show()
    
    plt.plot(temp[1100,:,:]) #osoba red br 122, svih 32 kanala #PSP
    plt.title("PSP CONV4")
    plt.show()
    
    ##
    temp = izlazi[19]
    plt.plot(temp[98,:,:]) #osoba red br 12, svih 32 kanala  #MSA
    plt.title("MSA, CONV5")
    plt.show()
    
    plt.plot(temp[555,:,:]) #osoba red br 122, svih 32 kanala #CTRL
    plt.title("CTRL CONV5")
    plt.show()
    
    plt.plot(temp[66,:,:]) #osoba red br 122, svih 32 kanala #PD
    plt.show()
    
    plt.plot(temp[1100,:,:]) #osoba red br 122, svih 32 kanala #PSP
    plt.title("PSP CONV5")
    plt.show()
    
    ##
    temp = izlazi[23]
    plt.plot(temp[98,:,:]) #osoba red br 12, svih 32 kanala  #MSA
    plt.title("MSA, CONV6")
    plt.show()
    
    plt.plot(temp[555,:,:]) #osoba red br 122, svih 32 kanala #CTRL
    plt.title("CTRL CONV6")
    plt.show()
    
    plt.plot(temp[66,:,:]) #osoba red br 122, svih 32 kanala #PD
    plt.title("PD, CONV6")
    plt.show()
    
    plt.plot(temp[1100,:,:]) #osoba red br 122, svih 32 kanala #PSP
    plt.title("PSP CONV6")
    plt.show()
    
    ## 
    #%
    
    fig = plt.figure(figsize = (20,20))
    
    temp = izlazi[3]
    
    plt.subplot(641)
    plt.plot(temp[98,0:200,8]) #osoba red br 12, svih 32 kanala  #MSA i 98 je neki msa izgleda nije strasan
    plt.plot(temp[98,0:200,12]) #osoba red br 12, svih 32 kanala  #MSA i 98 je neki msa izgleda nije strasan
    plt.plot(temp[98,0:200,28]) 
    plt.title("MSA")
    plt.ylabel("CONV1")
    
    plt.subplot(642)
    plt.plot(temp[555,0:200,8]) #osoba red br 122, svih 32 kanala #CTRL i 555 je ctrl mozda lepsa i 1130 je ok
    plt.plot(temp[555,0:200,12]) #osoba red br 122, svih 32 kanala #CTRL i 555 je ctrl mozda lepsa i 1130 je ok
    plt.plot(temp[555,0:200,28]) 
    plt.title("CTRL")
    
    plt.subplot(643)
    plt.plot(temp[66,0:200,8]) #osoba red br 122, svih 32 kanala #PD i 66 je pd i 77 ali cudan bas
    plt.plot(temp[66,0:200,12]) #osoba red br 122, svih 32 kanala #PD i 66 je pd i 77 ali cudan bas
    plt.plot(temp[66,0:200,28]) 
    plt.title("PD")
    
    plt.subplot(644)
    plt.plot(temp[1100,0:200,8]) #osoba red br 122, svih 32 kanala #PSP #imas i 211  a i 1100 nije strasno i 1120 mozda ok 
    plt.plot(temp[1100,0:200,12]) #
    plt.plot(temp[1100,0:200,28]) 
    plt.title("PSP")
    
    
    #kanal 8 je zanimljiv kod conv1
    # i 12 nesto fali kod msa
    #mozda i 13
    #i mozes 28
    
    ####
    temp = izlazi[7]
    plt.subplot(645)
    plt.plot(temp[98,0:200,8]) #osoba red br 12, svih 32 kanala  #MSA i 98 je neki msa izgleda nije strasan
    plt.plot(temp[98,0:200,12]) #osoba red br 12, svih 32 kanala  #MSA i 98 je neki msa izgleda nije strasan
    plt.plot(temp[98,0:200,28]) 
    plt.ylabel("CONV2")
    
    
    plt.subplot(646)
    plt.plot(temp[555,0:200,8]) #osoba red br 122, svih 32 kanala #CTRL i 555 je ctrl mozda lepsa i 1130 je ok
    plt.plot(temp[555,0:200,12]) #osoba red br 122, svih 32 kanala #CTRL i 555 je ctrl mozda lepsa i 1130 je ok
    plt.plot(temp[555,0:200,28]) 
    
    
    plt.subplot(647)
    plt.plot(temp[66,0:200,8]) #osoba red br 122, svih 32 kanala #PD i 66 je pd i 77 ali cudan bas
    plt.plot(temp[66,0:200,12]) #osoba red br 122, svih 32 kanala #PD i 66 je pd i 77 ali cudan bas
    plt.plot(temp[66,0:200,28]) 
    
    
    plt.subplot(648)
    plt.plot(temp[1100,0:200,8]) #osoba red br 122, svih 32 kanala #PSP #imas i 211  a i 1100 nije strasno i 1120 mozda ok 
    plt.plot(temp[1100,0:200,12]) #
    plt.plot(temp[1100,0:200,28]) 
    #####
    temp = izlazi[11]
    plt.subplot(649)
    plt.plot(temp[98,0:200,8]) #osoba red br 12, svih 32 kanala  #MSA i 98 je neki msa izgleda nije strasan
    plt.plot(temp[98,0:200,12]) #osoba red br 12, svih 32 kanala  #MSA i 98 je neki msa izgleda nije strasan
    plt.plot(temp[98,0:200,28]) 
    plt.ylabel("CONV3")
    
    
    plt.subplot(6,4,10)
    plt.plot(temp[555,0:200,8]) #osoba red br 122, svih 32 kanala #CTRL i 555 je ctrl mozda lepsa i 1130 je ok
    plt.plot(temp[555,0:200,12]) #osoba red br 122, svih 32 kanala #CTRL i 555 je ctrl mozda lepsa i 1130 je ok
    plt.plot(temp[555,0:200,28]) 
    
    
    plt.subplot(6,4,11)
    plt.plot(temp[66,0:200,8]) #osoba red br 122, svih 32 kanala #PD i 66 je pd i 77 ali cudan bas
    plt.plot(temp[66,0:200,12]) #osoba red br 122, svih 32 kanala #PD i 66 je pd i 77 ali cudan bas
    plt.plot(temp[66,0:200,28]) 
    
    
    plt.subplot(6,4,12)
    plt.plot(temp[1100,0:200,8]) #osoba red br 122, svih 32 kanala #PSP #imas i 211  a i 1100 nije strasno i 1120 mozda ok 
    plt.plot(temp[1100,0:200,12]) #
    plt.plot(temp[1100,0:200,28]) 
    
    ##############
    temp = izlazi[15]
    plt.subplot(6,4,13)
    plt.plot(temp[98,0:200,8]) #osoba red br 12, svih 32 kanala  #MSA i 98 je neki msa izgleda nije strasan
    plt.plot(temp[98,0:200,12]) #osoba red br 12, svih 32 kanala  #MSA i 98 je neki msa izgleda nije strasan
    plt.plot(temp[98,0:200,28]) 
    plt.ylabel("CONV4")
    
    
    plt.subplot(6,4,14)
    plt.plot(temp[555,0:200,8]) #osoba red br 122, svih 32 kanala #CTRL i 555 je ctrl mozda lepsa i 1130 je ok
    plt.plot(temp[555,0:200,12]) #osoba red br 122, svih 32 kanala #CTRL i 555 je ctrl mozda lepsa i 1130 je ok
    plt.plot(temp[555,0:200,28]) 
    
    
    plt.subplot(6,4,15)
    plt.plot(temp[66,0:200,8]) #osoba red br 122, svih 32 kanala #PD i 66 je pd i 77 ali cudan bas
    plt.plot(temp[66,0:200,12]) #osoba red br 122, svih 32 kanala #PD i 66 je pd i 77 ali cudan bas
    plt.plot(temp[66,0:200,28]) 
    
    
    plt.subplot(6,4,16)
    plt.plot(temp[1100,0:200,8]) #osoba red br 122, svih 32 kanala #PSP #imas i 211  a i 1100 nije strasno i 1120 mozda ok 
    plt.plot(temp[1100,0:200,12]) #
    plt.plot(temp[1100,0:200,28]) 
    
    ########3
    temp = izlazi[19]
    plt.subplot(6,4,17)
    plt.plot(temp[98,0:200,8]) #osoba red br 12, svih 32 kanala  #MSA i 98 je neki msa izgleda nije strasan
    plt.plot(temp[98,0:200,12]) #osoba red br 12, svih 32 kanala  #MSA i 98 je neki msa izgleda nije strasan
    plt.plot(temp[98,0:200,28]) 
    plt.ylabel("CONV5")
    
    
    plt.subplot(6,4,18)
    plt.plot(temp[555,0:200,8]) #osoba red br 122, svih 32 kanala #CTRL i 555 je ctrl mozda lepsa i 1130 je ok
    plt.plot(temp[555,0:200,12]) #osoba red br 122, svih 32 kanala #CTRL i 555 je ctrl mozda lepsa i 1130 je ok
    plt.plot(temp[555,0:200,28]) 
    
    
    plt.subplot(6,4,19)
    plt.plot(temp[66,0:200,8]) #osoba red br 122, svih 32 kanala #PD i 66 je pd i 77 ali cudan bas
    plt.plot(temp[66,0:200,12]) #osoba red br 122, svih 32 kanala #PD i 66 je pd i 77 ali cudan bas
    plt.plot(temp[66,0:200,28]) 
    
    
    plt.subplot(6,4,20)
    plt.plot(temp[1100,0:200,8]) #osoba red br 122, svih 32 kanala #PSP #imas i 211  a i 1100 nije strasno i 1120 mozda ok 
    plt.plot(temp[1100,0:200,12]) #
    plt.plot(temp[1100,0:200,28]) 
    #########3
    temp = izlazi[23]
    plt.subplot(6,4,21)
    plt.plot(temp[98,0:200,8]) #osoba red br 12, svih 32 kanala  #MSA i 98 je neki msa izgleda nije strasan
    plt.plot(temp[98,0:200,12]) #osoba red br 12, svih 32 kanala  #MSA i 98 je neki msa izgleda nije strasan
    plt.plot(temp[98,0:200,28]) 
    plt.ylabel("CONV6")
    
    
    plt.subplot(6,4,22)
    plt.plot(temp[555,0:200,8]) #osoba red br 122, svih 32 kanala #CTRL i 555 je ctrl mozda lepsa i 1130 je ok
    plt.plot(temp[555,0:200,12]) #osoba red br 122, svih 32 kanala #CTRL i 555 je ctrl mozda lepsa i 1130 je ok
    plt.plot(temp[555,0:200,28]) 
    
    
    plt.subplot(6,4,23)
    plt.plot(temp[66,0:200,8]) #osoba red br 122, svih 32 kanala #PD i 66 je pd i 77 ali cudan bas
    plt.plot(temp[66,0:200,12]) #osoba red br 122, svih 32 kanala #PD i 66 je pd i 77 ali cudan bas
    plt.plot(temp[66,0:200,28]) 
    
    
    plt.subplot(6,4,24)
    plt.plot(temp[1100,0:200,8]) #osoba red br 122, svih 32 kanala #PSP #imas i 211  a i 1100 nije strasno i 1120 mozda ok 
    plt.plot(temp[1100,0:200,12]) #
    plt.plot(temp[1100,0:200,28]) 
    
    fig.tight_layout()
    plt.show()