EN.705.603.81.FA23 - CREATING AI ENABLED SYSTEMS
MODULE 04 - ASSIGNMENT
Created by: Antonio Armenta
Date: September 24, 2023

In this repository you will find:
* "cars.csv"
  Input data regarding different car features and the duration in days that each car was listed until it was sold.
  Relevant car features include transmision, color, odometer_value, year_produced, body_type, & prices_usd.

* carsfactors.py
  A class with two main components: a model learn and a model inference.
  The model learn function processes the input data, employes one ordinal encoder and two onehot encoders, and scales the data.
  Then, the function is trained using the processed data and linear regression.
  
  The model infer function will take inputs, reprocess them using the learned encoders and make a linear regression inference.
  The output is the expected number of days a car will be listed based on the input parameters.
  
* carfactors_service.py
  Defines a flask service to get stats and input parameters (arguments) to make inferences using the carsfactors class
  
* carfactors.ipynb
  A notebook that can be used to run and test the model.
  In the "Get Determination" section, you will be able to type in the input parameters for the inference.
  When you run the notebook you will get a hyperlink that includes the arguments for the flask app. Upon clicking, the browser will display the inferrence result.
  There is also a summary and recommendations sections at the end of this notebook.

