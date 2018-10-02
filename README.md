# FingerTapping


Collected IMU and FSR data during finger tapping test from patients with neurological disorders and healthy controls.
Working on computer-aided differential diagnostics. 

To get a prediction for an unseen piece of data, start a local server and load model by running "ServeModel.py" (flask app).
Navigate to the local server in your browser and choose a file (well, this is all very much local and the path of the directory is hard coded. Sorry.) Submit.

Alternatively, the whole modelling process goes like:
1. Run TappingSaveReshappedData.ipynb
2. Run TappingML.ipynb 
3. Run PredictTestDescriptive.ipynb

TappingForceExamine.ipynb is standalone and not really relevant at the moment. So is dropout plot. 

All of these scripts also require some minor modification to run, particularly setting the working directory or choosing what samples to predict.





