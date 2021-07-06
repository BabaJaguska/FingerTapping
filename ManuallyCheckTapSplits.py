# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 16:05:51 2020

@author: minja
"""

from readAndEncode import readAllDataAndAutoSplit, readAllDataAndSplitFromTxt
import argparse
import matplotlib
from threading import Thread
import json
import numpy as np
import os
from matplotlib import pyplot as plt
matplotlib.use('Qt5Agg')



#%%

def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target = fn, args = args, kwargs = kwargs)
        thread.daemon = True
        thread.start()
        return thread
    return wrapper

class dataPlotter:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(2,1)
        self.mes_idx = -1
        self.connect_keypress_event()
        self.connect_fig_close_event()
        
    def fig_close_handler(self, event):
        print('CLOSING')
    
    def fig_keypress_handler(self, event):
        
        # NEXT PLOT 
        if event.key == 'n':   
            if self.mes_idx < len(self.data):
                print('PROCEEDING TO NEXT PLOT')
                self.mes_idx += 1
                self.plotData()
                self.draw()

            else:
                print('ALL DONE!')
            
        if event.key == 'm':
            print('=========================================================')
            print('Modifying current segmentation points')
            print('=========================================================')
            self.getUserInput()
            
        if event.key == 'q':
            print('oops?')
            
            
    def connect_fig_close_event(self):
        self.cid = self.fig.canvas.mpl_connect('close_event', self.fig_close_handler)
        
    def connect_keypress_event(self):
        self.kid = self.fig.canvas.mpl_connect('key_press_event', self.fig_keypress_handler)

        
    def beginPlotting(self, data):
        self.data = data
        plt.show(block=True)
        self.plotData()
        
        
    #@threaded
    def plotData(self):
        
        temp = self.data[self.mes_idx]
        mes = temp['measurement']
        intermediate_signal = temp['intermediate_signal']
        peak_indices = temp['peak_indices']
        fname = mes.id
            
        self.ax[0].cla()
        
        self.ax[0].plot(intermediate_signal)
        self.ax[0].plot(peak_indices, intermediate_signal[peak_indices], 'r*')
        
        
        self.ax[1].cla()
        #self.ax[1].plot(mes.gyro1xT)
        #self.ax[1].plot(mes.gyro1yT)
        #self.ax[1].plot(mes.gyro1zT)
        self.ax[1].plot(mes.gyro2xT)
        #self.ax[1].plot(mes.gyro2xT)
        #self.ax[1].plot(mes.gyro2xT)
        markerline, stemline, baseline = self.ax[1].stem(peak_indices, 
                                                         [10]*len(peak_indices),
                                                         ':' ,use_line_collection = True)
        plt.setp(markerline, markersize = 1)
        plt.title(fname)
        
        
    def draw(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    @threaded
    def getUserInput(self):
        
        measurementID = self.data[self.mes_idx]['measurement'].id
        print(measurementID)
        
        pointsToAdd = []
        pointsToSubtract = []
        while True:
            print('\na-Add points\ns-Subtract points\nx-Exit\n')
            action = input("What would you like to do? (a|s|x): ")
            if action == 'x':
                print("Finished modifying. Return to the plot figure and press 'n' to inspect the next plot.")
                print("Close the figure or press 'q' to finish all.")
                break
            
            if action == 'a':
                
                while True:
                    pointToAdd = input('Input x coordinate to add and press enter. To quit press "x"\n')
                    if pointToAdd.lower() == 'x':
                        break                    
                    try:
                        pointsToAdd.append(int(pointToAdd))
                    except:
                        print('>>>>> Invalid entry. <<<<<')
                    
            if action == 's':
                while True:
                    pointToSubtract = input('Input x coordinate to subtract and press enter. To quit press "x"\n')
                    if pointToSubtract.lower() == 'x':
                        break
                    try:
                        pointsToSubtract.append(int(pointToSubtract))
                    except:
                        print('>>>>> Invalid entry. <<<<<')
                        
            if action not in ['a', 's', 'x']:
                print('>>>>> Invalid entry. <<<<<')
        
        modifier = {"id": measurementID,
                    "add":pointsToAdd,
                    "sub": pointsToSubtract}
        
        with open('./ModifyPoints.txt', 'a') as f:
            print(json.dumps(modifier),file = f)
                
        
        return


def addPoints(arr, points):
    points = [point for point in points if point not in arr]
    arr = np.append(arr, points)    
    arr = np.sort(arr)
    return arr

def removePoints(arr, roughPoints):
    
    def closest(arr, point):
        
        if point in arr:
            return np.where(arr == point)
        
        for pointCandidate in range(point - 3, point + 3):
            if pointCandidate in arr:
                return np.where(arr == pointCandidate)
            
        print('POINT {} NOT FOUND.'.format(point))
        
        return -1
    
    point_indices = [closest(arr, point) for point in roughPoints]     
    
    arr = np.delete(arr, point_indices)  
    
    return arr

def modifySignalSplits(splitPoints, modifyDict):
    
    if modifyDict['add']:    
        splitPoints = addPoints(splitPoints, modifyDict['add'])
        
    if modifyDict['sub']:
        splitPoints = removePoints(splitPoints, modifyDict['sub'])
    
    return splitPoints

        
def findMeasurementByID(data, ID):
    
    for idx, datum in enumerate(data):
        if datum['measurement'].id == ID:
            return datum, idx
        
    print('COULD NOT FIND MEASUREMENT')
    return

def writeAllSignalSplitsToFile(data, filename):
    
    if os.path.exists(filename):
        print('REWRITING FILE !!!')           
    
    with open(filename, 'w+') as f:
        for datum in data:
            temp = {'id': datum['measurement'].id,
                    'allSplitPoints': datum['peak_indices'].tolist()}
            print(json.dumps(temp), file = f)

            
    return

def modifyAutoSplits(allData, modifierFile, fileToWrite):
    modifiers = []
    try:
        with open(modifierFile, 'r') as f:
            lines = f.readlines()
            for line in lines:
                modifyFile = json.loads(line)
                modifiers.append(modifyFile)
    except:
        print('Could not read the requested file.')
    
    # print(modifiers)
    
    # ========= MODIFY SPLIT POINTS ========
    for modifierDict in modifiers:
        print('Modifying ', modifierDict['id'])
        tempDatum, idx = findMeasurementByID(allData, modifierDict['id'])
        print(idx)
        tempSplits = tempDatum['peak_indices']
        allData[idx]['peak_indices'] = modifySignalSplits(tempSplits, modifierDict)
        
    
    writeAllSignalSplitsToFile(allData, fileToWrite)

    return 
        
#%%

#TODO: prvo se pojavi prazan plot, mora da se klikne 'n' kao next, popravi ovo!

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description = 'Correct automatically determined tap boundaries')
    arg_parser.add_argument('-d', '--dataPath', help = 'Your recordings path')
    arg_parser.add_argument('-m', '--method', help = 'Auto or file')
    arg_parser.add_argument('-f','--file', help = 'Which file to read or write')
    args = arg_parser.parse_args()
    
    dataPath = args.dataPath
    splitMethod = args.method
    filename = args.file
    
    if dataPath is None:
        # dataPath = r'C:\data\icef\tapping\raw data'
        dataPath = './data/raw data1/'
        
    if splitMethod is None:
        splitMethod = 'auto'
        
    # ======= READ DATA =======
    if splitMethod == 'auto':
        integrateFirst = 1
        print('===========================================================')
        print('Press N to continue to next plot')
        print('Press M to modify the points')
        print('Press X to stop editing')
        print('Press Q to quit')
        print('===========================================================')
        allData = readAllDataAndAutoSplit(dataPath, integrateFirst, filename)
        
    elif splitMethod == 'file':
        
        allData = readAllDataAndSplitFromTxt(dataPath, filename, integrateFirst)
    else:
        raise('Invalid split method. Choose between "auto" and "file"')            
                   
    # ======= PLOT =======
    gui = dataPlotter()       
    gui.beginPlotting(allData) 
    
