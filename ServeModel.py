# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 14:38:13 2018

@author: MinjaBelic
"""
# %% import libarries

import numpy as np
import flask
from keras.models import load_model
import scipy.io

#basic app. run from conda with: python ServeModel.py then run SimpleRequest.py for now



# %%initialize Flask app
app = flask.Flask(__name__)
root = 'C:/Users/Korisnik/Desktop/Minjino/TAPPING/'

def predictSignal(signalX, model):
    pred = {}
    signalX = np.expand_dims(signalX,axis=0)
    prediction = model.predict(signalX)
    prediction = prediction[0]
    prediction = np.round(np.float64(prediction),2)
    
    d = {0: 'CTRL',
        1: 'MSA',
        2: 'PD',
        3: 'PSP'}
    
    

    #pred['actualDiagnosis'] = d[np.argmax(signalY)]
    pred['predictedDiagnosis'] = d[np.argmax(prediction)]
    pred['CTRL'] = prediction[0]
    pred['MSA'] = prediction[1]
    pred['PD'] = prediction[2]
    pred['PSP'] = prediction[3]
     
    
    return pred

#print(model.summary())
def loadModel(root):
    global model
    model = load_model(root + 'my_model.h5')
    model._make_predict_function()
    model.load_weights(root+'CNN25sep4.25.9.2018.15.5valjda.h5')
    return

# %% test route

@app.route("/bratac",methods = ["GET"])

def bratac():
    return 'MWHAHAAH BRT'
# %% actual route

@app.route("/predict",methods = ["POST"])

def predict():
    data = {"success":False}
      
    if flask.request.method == "POST":
        signal = flask.request.files["signal"]
        print(signal)
        print('aman')
        signal = scipy.io.loadmat(signal)
        print(signal)
        signal = signal['X']
       
        pred = predictSignal(signal,model)
            # make sure to be able to accept raw signals and then prepare them
        
        data["predictions"] = pred
        data["success"] = True
        print(data)
            
    return flask.jsonify(data)

# %% wait            

if __name__=="__main__":
    # if this is the main thread of execution first load the model and then start server
    # call to loadModel() is a blocking operation 
    # prevents the web service from starting before the model is fully loaded 
    print("Loading model and Flask starting server. Please wait...")
    loadModel(root)
    app.run(debug=True)



