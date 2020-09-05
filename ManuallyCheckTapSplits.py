# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 16:05:51 2020

@author: minja
"""

from measurement import measurement
from readAndEncode import *
from time import sleep




#=======================================
root = r'C:\data\icef\tapping\raw data'

#%%

class dataPlotter:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(2,1)
        self.mes_idx = 0
        self.connect_keypress_event()
        self.connect_fig_close_event()
        
    def beginPlotting(self, data):
        self.data = data
        self.plotData()
        
    def plotData(self):
        
        temp = self.data[self.mes_idx]
        mes = temp['measurement']
        intermediate_signal = temp['intermediate_signal']
        peak_indices = temp['peak_indices']
            
        self.ax[0].cla()
        
        self.ax[0].plot(intermediate_signal)
        self.ax[0].plot(peak_indices, intermediate_signal[peak_indices], 'r*')
        
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
            
        
    def connect_fig_close_event(self):
        self.fig.canvas.mpl_connect('close_event', self.fig_close_handler)
        
    def connect_keypress_event(self):
        self.fig.canvas.mpl_connect('key_press_event', self.fig_keypress_handler)

        
    


#%%

allPatientFolders, allPatientDiagnoses = getUniquePatients(root)


allData = []


for patientFolder in allPatientFolders:
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
        
####
#%%
        
gui = dataPlotter()     
gui.beginPlotting(allData[1:3])  
        

            

        