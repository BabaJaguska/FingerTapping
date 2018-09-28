# FingerTapping


Collected IMU and FSR data during finger tapping test from patients with neurological disorders and healthy controls.
Working on computer-aided differential diagnostics. 


TappingForceExamine is standalone and not really relevant at the moment. So is dropout plot.

1. Run TappingSaveReshappedData
2. Run TappingML 
3. Run PredictTestDescriptive

3a.Alternatively start a local server and load model by running ServeModel.py 
and then predict a single file through SimpleRequest.py (doesn't accept arguments yet.)


All of these scripts require some minor modification to run, particularly setting the working directory or choosing what samples to predict.





