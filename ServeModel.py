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
import requests

#basic app. run from conda with: python ServeModel.py then run SimpleRequest.py for now



# %%initialize Flask app
app = flask.Flask(__name__)
root = 'C:/Users/Korisnik/Desktop/Minjino/TAPPING/'
signalsRoot = 'C:/Users/Korisnik/Desktop/Minjino/TAPPING/TEST_FILES/'

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
    model = load_model(root + 'MODELCNN16Batch9s9KERNEL9CEO.h5')
    model._make_predict_function()
    model.load_weights(root+'!BEST_WEIGHTSCNN16Batch9s9KERNEL9.10.10.2018.9.20.h5')
    return
# %% index
    
@app.route("/")
def basic():
    return flask.redirect(flask.url_for('index'))
  
# %% test route
#
#@app.route("/bratac",methods = ["GET"])
#
#def bratac():
#    return 'MWHAHAAH BRT'

# %% get file name to post

@app.route("/_getFileName",methods = ["POST","GET"])
def getFileName():
#    
    try:
        fileName = flask.request.args.get('fileName',type=str)
        fileName = fileName.split("\\").pop();
        fileName = signalsRoot + fileName
       
        signal = open(fileName,'rb').read(); ## loool? ovo treba zasto?

        payload = {"signal":signal}
        response = requests.post('http://localhost:5000/_PREDICT', files = payload).json()
        print(response['success'])
        if response['success']:
            return flask.jsonify(response['predictions'])
        else:
            return flask.jsonify("Failed request")
    except Exception as e:
        return(str(e))
        
# %% hm interactive stuff
@app.route("/index",methods = ["POST","GET"])
def index():
    try:
        return flask.render_template("index.html")
    except Exception as e:
        return(str(e))
        
# %% actual route

@app.route("/_PREDICT",methods = ["POST"])

def _PREDICT():
    data = {"success":False}
    
    if flask.request.method == "POST":
        signal = flask.request.files["signal"]
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



