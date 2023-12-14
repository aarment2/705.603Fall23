# IVF Processing
# An in-depth analysis of features characterizing IVF procedures and success rates in the United States.
# The datasets are from the 2021 US Department of Health report, which is the latest available as of 12/12/2023.
# Data from previous years can be processed using this same code.

# Importing the libraries
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from IPython.display import display


class IVF():

    #Initialization
    def __init__(self):
        self.patient_cycles_df = pd.DataFrame()
        self.services_df = pd.DataFrame()
        self.success_rates_df = pd.DataFrame()
        self.summary_df = pd.DataFrame()
        self.compound_df = pd.DataFrame()
        self.results = ''
        self.modelLearn = False
        self.stats = 0


    #This function receives a dataframe and a question id. The question id will be filtered out from the dataframe rows and then converted into a feature (column) of the dataframe
    #Matching the columns from the original array
    def merge_data_value_num(self, reference_df, question_id):

            suffix = '_'+question_id
            temp_df = reference_df[reference_df['QuestionId'] == question_id]

            merged_df = pd.merge(reference_df, temp_df[['ClinicId', 'FilterId', 'BreakOutCategoryId', 'BreakOutId', 'Data_Value_num']],
                         on=['ClinicId', 'FilterId', 'BreakOutCategoryId', 'BreakOutId'], how='left', suffixes=('', suffix))
            
            merged_df.drop_duplicates(inplace=True)

            return merged_df
              
    
    #This function performs all of the data ingest. It reads four csv dataset files, cleans them up, removes non-relevant features, reorganizes it, and compounds a final dataset for training and testing.
    def ingest(self):
        
        #Read the csv datasets, validating files exist and checking for other errors.
        dataset_directory = 'datasets/'
        try: 
            patient_cycles_df = pd.read_csv(dataset_directory + '2021_patient_cycles.csv')
            services_df = pd.read_csv(dataset_directory + '2021_services.csv')
            success_rates_df = pd.read_csv(dataset_directory + '2021_success_rates.csv', low_memory=False)
            summary_df = pd.read_csv(dataset_directory + '2021_summary.csv', low_memory=False)
            self.results = 'CSV files succesfully converted to dataframes.<br>'
            print("CSV files succesfully converted to dataframes.")
        except FileNotFoundError as e:
            self.results = 'File not found'
            print("File not found:", e)
        except Exception as e:
            self.results = 'An error occurred'
            print("An error occurred:", e)
        
        
        ##### Patient Cycles Dataset #####
        #Remove unnecessary features
        columns_to_drop = ['Year', 'LocationAbbr', 'LocationDesc', 'FacilityName', 'MedicalDirector', 'Address', 'City', 'ZipCode', 'Phone', 
                           'Clinic Status', 'Geolocation', 'Topic', 'Question', 'Breakout_Category', 'Breakout', 'Data_Value', 'Data_Value_Footnote_Symbol', 'Data_Value_Footnote']
        patient_cycles_df.drop(columns=columns_to_drop, inplace=True)
        
        #Reorganize feature sequence
        column_seq = ['ClinicId', 'TopicId', 'QuestionId', 'BreakOutCategoryId', 'BreakOutId', 'Data_Value_num', 'Cycle_Count']
        patient_cycles_df = patient_cycles_df[column_seq]
        
        #Move ClinicId to the first position
        column_to_move = 'ClinicId'
        patient_cycles_df = patient_cycles_df[ [column_to_move] + [col for col in patient_cycles_df.columns if col != column_to_move] ]
        
        #Replace all NaNs with zeroes
        patient_cycles_df.fillna(0, inplace=True)
        
        
        
        ##### Services Dataset #####
        #Remove unnecessary features
        columns_to_drop = ['Year', 'LocationAbbr', 'LocationDesc', 'FacilityName', 'MedicalDirector', 'Address', 'City', 'Zipcode', 'Phone', 
                           'Clinic Status', 'Geolocation', 'Topic', 'SubTopic']
        services_df.drop(columns=columns_to_drop, inplace=True)
        
        #Reorganize feature sequence
        column_seq = ['ClinicId', 'TopicId', 'SubTopicId', 'Data_Value']
        services_df = services_df[column_seq]
        
        #Move ClinicId to the first position
        column_to_move = 'ClinicId'
        services_df = services_df[ [column_to_move] + [col for col in services_df.columns if col != column_to_move] ]
        
        #Replace all NaNs with zeroes
        services_df.fillna(0, inplace=True)
        
       

        ##### Success Rates Dataset #####
        #Remove unnecessary features
        columns_to_drop = ['Year', 'LocationAbbr', 'LocationDesc', 'FacilityName', 'MedicalDirector', 'Address', 'City', 'ZipCode', 'Phone', 'Cycle_Count', 'Data_Value', 'TypeId',
                           'Clinic Status', 'GeoLocation', 'Type', 'Topic', 'Question', 'Filter', 'Breakout_Category', 'Breakout', 'Data_Value_Footnote_Symbol', 'Data_Value_Footnote']
        success_rates_df.drop(columns=columns_to_drop, inplace=True)
        
        
        #Move ClinicId to the first position
        column_to_move = 'ClinicId'
        success_rates_df = success_rates_df[ [column_to_move] + [col for col in success_rates_df.columns if col != column_to_move] ]
               
        #Remove rows with questions not needed
        success_rates_df = success_rates_df.drop(success_rates_df[success_rates_df['QuestionId'] == 'Q012'].index)
        success_rates_df = success_rates_df.drop(success_rates_df[success_rates_df['QuestionId'] == 'Q013'].index)
        success_rates_df = success_rates_df.drop(success_rates_df[success_rates_df['QuestionId'] == 'Q014'].index)
        success_rates_df = success_rates_df.drop(success_rates_df[success_rates_df['QuestionId'] == 'Q016'].index)
        success_rates_df = success_rates_df.drop(success_rates_df[success_rates_df['QuestionId'] == 'Q017'].index)
        success_rates_df = success_rates_df.drop(success_rates_df[success_rates_df['QuestionId'] == 'Q018'].index)
        success_rates_df = success_rates_df.drop(success_rates_df[success_rates_df['QuestionId'] == 'Q020'].index)
        success_rates_df = success_rates_df.drop(success_rates_df[success_rates_df['QuestionId'] == 'Q021'].index)
        success_rates_df = success_rates_df.drop(success_rates_df[success_rates_df['QuestionId'] == 'Q022'].index)
        success_rates_df = success_rates_df.drop(success_rates_df[success_rates_df['QuestionId'] == 'Q023'].index)
        success_rates_df = success_rates_df.drop(success_rates_df[success_rates_df['QuestionId'] == 'Q028'].index)
        success_rates_df = success_rates_df.drop(success_rates_df[success_rates_df['QuestionId'] == 'Q029'].index)
        success_rates_df = success_rates_df.drop(success_rates_df[success_rates_df['QuestionId'] == 'Q031'].index)
        success_rates_df = success_rates_df.drop(success_rates_df[success_rates_df['QuestionId'] == 'Q032'].index)
        success_rates_df = success_rates_df.drop(success_rates_df[success_rates_df['QuestionId'] == 'Q033'].index)

        #Converting some rows / questions into features.         
        success_rates_df = self.merge_data_value_num(success_rates_df, 'Q024')
        success_rates_df = self.merge_data_value_num(success_rates_df, 'Q025')
        success_rates_df = self.merge_data_value_num(success_rates_df, 'Q026')
        success_rates_df = self.merge_data_value_num(success_rates_df, 'Q027')

        #Reorganize feature sequence
        column_seq = ['ClinicId', 'TopicId', 'QuestionId', 'FilterId', 'BreakOutCategoryId', 'BreakOutId', 'Data_Value_num_Q024', 'Data_Value_num_Q025', 'Data_Value_num_Q026', 'Data_Value_num_Q027', 'Data_Value_num']
        success_rates_df = success_rates_df[column_seq]       

        #Rename converted features
        success_rates_df.rename(columns={'Data_Value_num_Q024': 'avg_intended'}, inplace=True)
        success_rates_df.rename(columns={'Data_Value_num_Q025': 'first_time_success'}, inplace=True)
        success_rates_df.rename(columns={'Data_Value_num_Q026': 'retry_success'}, inplace=True)
        success_rates_df.rename(columns={'Data_Value_num_Q027': 'all_success'}, inplace=True)
        success_rates_df.rename(columns={'Data_Value_num': 'global_success'}, inplace=True)
        
        #Replace all NaNs with zeroes. In this dataset, NaNs are created due to percentages between 0 and 1% annotated as strings. It's correct to replace with zeroes.
        success_rates_df.fillna(0, inplace=True)
        
        
        ##### Summary Dataset #####
        #Remove unnecessary features
        columns_to_drop = ['Year', 'LocationAbbr', 'LocationDesc', 'FacilityName', 'MedicalDirector', 'Address', 'City', 'ZipCode', 'Phone', 'DisplayOrder', 'Clinic Status',
                           'Geolocation', 'Topic', 'SubTopic', 'Question', 'Breakout_Category', 'Breakout', 'Data_Value', 'Data_Value_Footnote_Symbol', "Data_Value_Footnote"]
        summary_df.drop(columns=columns_to_drop, inplace=True)
        
        #Reorganize feature sequence
        column_seq = ['ClinicId', 'TopicId', 'SubTopicId', 'QuestionId', 'BreakoutCategoryId', 'BreakoutId', 'data_value_num', 'Cycle_Count']
        summary_df = summary_df[column_seq]
                       
        #Move ClinicId to the first position
        column_to_move = 'ClinicId'
        summary_df = summary_df[ [column_to_move] + [col for col in summary_df.columns if col != column_to_move] ]
        
        #Replace all NaNs with zeroes
        summary_df.fillna(0, inplace=True)
        
        
        # Define the file path for the output CSV file
        output_file_path = 'output_data.csv'

        # Export the updated DataFrame to a CSV file
        #Uncomment this if you would like to create a csv file for a desired dataframe.
        #success_rates_df.to_csv(output_file_path, index=False)
        
        
        #Assign ingested dataframes to the global dataframes
        self.patient_cycles_df = patient_cycles_df
        self.services_df = services_df
        self.summary_df = summary_df
        self.success_rates_df = success_rates_df
        self.compound_df = success_rates_df
    
        self.results = self.results + 'Data ingest completed succesfully.<br>'
        print("Data ingest completed succesfully.")

    
    #This function takes a compound dataframe and applies three models of regression: Linear, Random Forest, and Gradient Boosting.
    #The three regression models are compared and evaluated (See notebook)
    def model_learn(self):
        X = self.compound_df.drop(['global_success'], axis=1)
        y = self.compound_df['global_success']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        #Label Encoding
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
        for col in X_train_encoded.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X_train_encoded[col] = le.fit_transform(X_train_encoded[col])
        for col in X_test_encoded.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X_test_encoded[col] = le.fit_transform(X_test_encoded[col])       


        #Scaling                
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_encoded)
        X_test_scaled = scaler.fit_transform(X_test_encoded)
        y_train_scaled = y_train
        y_test_scaled = y_test

        
        #Linear Regression
        regressor1 = LinearRegression()
        regressor1.fit(X_train_scaled, y_train_scaled)
        accuracy1 = regressor1.score(X_train_scaled, y_train_scaled)
        test_accuracy1 = regressor1.score(X_test_scaled, y_test_scaled)

        self.results = self.results + 'Linear Regression: Accuracy on Training Data: '+str(accuracy1)+'<br>'
        self.results = self.results + 'Linear Regression: Accuracy on Test Data: '+str(test_accuracy1)+'<br>'
        print("Linear Regression: Accuracy on Training Data: ", accuracy1)
        print("Linear Regression: Accuracy on Test Data: ", test_accuracy1)
        print('\n')

        #Random Forest Regression
        #regressor2 = RandomForestRegressor()
        
        #Array of parameters for random search of best parameters set
        #Number of trees in the forest, maximum depth, minimum sample split, minimum number of samples per leaf, number of features per split, selection method
        #param_grid = {'n_estimators': [100, 500, 1000], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'max_features': ['auto', 'sqrt'], 'bootstrap': [True, False] }      
              
        #Best parameters search
        #random_search = RandomizedSearchCV(estimator=regressor2, param_distributions=param_grid, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)       
        #random_search.fit(X_train_scaled, y_train)
        #best_params = random_search.best_params_
        #best_score = random_search.best_score_
        
        #print("Best Hyperparameters:", best_params)
        #print("Best Score:", best_score)
        
        #Model fit using the best parameters
        #best_params = { 'n_estimators': 500, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'bootstrap': True }
        #regressor2 = RandomForestRegressor(**best_params)
        
        regressor2 = RandomForestRegressor()
        regressor2.fit(X_train_scaled, y_train_scaled)
        accuracy2 = regressor2.score(X_train_scaled, y_train_scaled)
        test_accuracy2 = regressor2.score(X_test_scaled, y_test_scaled)
                
        self.results = self.results + 'Random Forest: Accuracy on Training Data: '+str(accuracy2)+'<br>'
        self.results = self.results + 'Random Forest: Accuracy on Test Data: '+str(test_accuracy2)+'<br>'
        print("Random Forest: Accuracy on Training Data: ", accuracy2)
        print("Random Forest: Accuracy on Test Data: ", test_accuracy2)
        print('\n')

        #Gradient Boosting Regressor
        regressor3 = GradientBoostingRegressor()
        regressor3.fit(X_train_scaled, y_train)
        accuracy3 = regressor3.score(X_train_scaled, y_train)
        test_accuracy3 = regressor3.score(X_test_scaled, y_test)

        self.results = self.results + 'Gradient Boosting Regressor: Accuracy on Training Data: '+str(accuracy3)+'<br>'
        self.results = self.results + 'Gradient Boosting Regressor: Accuracy on Test Data: '+str(test_accuracy3)+'<br>'
        print("Gradient Boosting Regressor: Accuracy on Training Data: ", accuracy3)
        print("Gradient Boosting Regressor: Accuracy on Test Data: ", test_accuracy3)
        print('\n')











