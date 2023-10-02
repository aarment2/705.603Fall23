# NLP
EN.705.603.81.FA23 - Creating AI Enabled Systems
MODULE 5 - ASSIGNMENT
October 01, 2023
Antonio Armenta

This module is an introduction to Natural Language Processing
natural_language_processing.py contains three main functions:
* _cleanup
    Gets the input data from the tsv file and splits it and converts it into several arrays
    After that, pertinent stopwords are filtered out

* model_train
    This function splits the data into inputs (X) and output (Y), then it also splits it into training and validating sets
    After that, the model is trained using the Naives Bayes model

* model_infer
    This function gets a new string passed as an input, then makes an inference based upon the trained model

natural_language_processing_Service.py helps us review model stats and make inferences as using the GET function

This code was tested using the NLPProcessing.ipynb notebook