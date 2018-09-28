# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 15:04:24 2018

@author: Korisnik
"""

# %% consume the API programatically

import requests

#initialize endpoint URL with the input file path
KERAS_REST_API_URL = "http://localhost:5000/predict"
root = 'C:/Users/Korisnik/Desktop/Minjino/TAPPING/'
FILE_NAME = "TestSample.mat"
signal = root+FILE_NAME
signal = open(signal,'rb').read() ## loool? ovo treba zasto?

payload = {"signal":signal}


r = requests.post(KERAS_REST_API_URL, files = payload).json()

if r["success"]:
    print(r['predictions'])
else:
    print("Request failes")
