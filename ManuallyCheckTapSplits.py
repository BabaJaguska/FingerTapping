# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 16:05:51 2020

@author: minja
"""

from measurement import measurement
from readAndEncode import *
from time import sleep
import argparse
import matplotlib
from threading import Thread
import json
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
        self.mes_idx = 0
        self.connect_keypress_event()
        self.connect_fig_close_event()
        
    def fig_close_handler(self, event):
        print('CLOSING')
    
    def fig_keypress_handler(self, event):
        
        # NEXT PLOT 
        if event.key == 'n':   
            if self.mes_idx < len(self.data):
                print('PROCEEDING TO NEXT PLOT')
                self.plotData()
                self.draw()
                self.mes_idx += 1
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
        plt.title(fname)
        
        self.ax[1].cla()
        self.ax[1].plot(mes.gyro1xT)
        self.ax[1].plot(mes.gyro1yT)
        self.ax[1].plot(mes.gyro1zT)
        self.ax[1].plot(mes.gyro2xT)
        self.ax[1].plot(mes.gyro2xT)
        self.ax[1].plot(mes.gyro2xT)
        markerline, stemline, baseline = self.ax[1].stem(peak_indices, [10]*len(peak_indices), ':' ,use_line_collection = True)
        plt.setp(markerline, markersize = 1)
        
        
        
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






        
    




        
#%%

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description = 'Correct automatically determined tap boundaries')
    arg_parser.add_argument('-d', '--dataPath')
    args = arg_parser.parse_args()
    dataPath = args.dataPath
    
    if dataPath is None:
        dataPath = r'C:\data\icef\tapping\raw data'
        
    print('Reading all relevant data...\n')     
  

    
    # READ DATA 
    allPatientFolders, allPatientDiagnoses = getUniquePatients(dataPath)
    
    allData = []
    
    for patientFolder in tqdm(allPatientFolders):
        currentPatientMeasurements = readPatientFiles(patientFolder)
        for mes in currentPatientMeasurements:
            if not mes.isRightHand():
                continue
            
            temp = {}
            intermediate_signal, peak_indices, _, _ = mes.splitTaps()
            temp['intermediate_signal'] = intermediate_signal
            temp['peak_indices'] = peak_indices
            temp['measurement'] = mes
            allData.append(temp)
        
    # ======= PLOT =======
    gui = dataPlotter()       
    gui.beginPlotting(allData) 
    
    #==== CHECK SAVED ====
    modifiers = []
    with open('./ModifyPoints.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            modifyFile = json.loads(line)
            modifiers.append(modifyFile)

    print(modifiers)        