TIME SERIES ASSIGNMENT
MODULE 6
Antonio Armenta


This program consists of two classes:
- time_series_processing.py
        This class uses two files as inputs to train the models: carotid_pressure.csv and illiac_pressure.csv
        The following three models are trained: 1) Carotid Pressure Model, 2) Illiac Pressure Model, 3) Combined Model.
- time_series_processing_Service.py
        This class integrates a web service to test the models.
        To test the model, please use the following HTML strings.
        
        
        ***** COMBINED MODEL ******
        Accuracy of the combined model:
        http://localhost:8786/stats_combined
        
        Input two files for inference using the combined model (carotid and illiac):     
        http://localhost:8786/infer?carotid_timeseries_filename=/FULL/PATH/TO/TIME/SERIES/FILE/carotid_pressure_test_1.csv&illiac_timeseries_filename="/FULL/PATH/TO/TIME/SERIES/FILE/illiac_pressure_test_1.csv
        Example:
        http://localhost:8786/infer_combined?carotid_timeseries_filename=data/carotid_pressure_test_3.csv&illiac_timeseries_filename=data/illiac_pressure_test_3.csv
        
        
        ***** CAROTID MODEL ******
        Accuracy of the carotid model:
        http://localhost:8786/stats_carotid
        
        Input one file for inference using the carotid model (example):
        http://localhost:8786/infer_carotid?carotid_timeseries_filename=data/carotid_pressure_test_3.csv
        
        
        
        ***** ILLIAC MODEL ******
        Accuracy of the illiac model:
        http://localhost:8786/stats_illiac
        
        Input one file for inference using the illiac model (example):
        http://localhost:8786/infer_illiac?illiac_timeseries_filename=data/illiac_pressure_test_3.csv