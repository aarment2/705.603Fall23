# Time Series Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from tsfresh import extract_features


class CombinedCardiacPressure():
    def __init__(self):
        self.modelLearn = False
        self.stats = 0
        self.scaler_carotid = StandardScaler()
        self.scaler_illiac = StandardScaler()

        
    def _cleanup(self, data, sigma=1):
        # interpolate data
        data = data.interpolate(method='linear', axis=1)
        data = data.interpolate(method="bfill", axis=1) #Replace all remaining NaNs

        # filter noise (use scipy.ndimage.gaussian_filter1d) 
        # estimate the standard deviation (sigma)
        sigma_row = data.std(axis=1)
        if data.shape[0] > 1:      
            sigma_avg = np.mean(sigma_row)
        else:
            sigma_avg = sigma_row[0]
        
        rows=0
        for index,row in data.iterrows():
            data.iloc[rows,:] = scipy.ndimage.gaussian_filter1d(data.iloc[rows,:], sigma_avg)
            rows += 1

        return data   
    
    
    def _removeNans(self, data):
        #All rows with more than 100 NaNs are removed. 100 NaNs equals ~30% of missing data for that row. The maximum allowed per best practices.
        #Roughly 12% of all rows have 30% of missing data (more than 100 NaNs). Removing them, increased the accuracy to ~70%
        rows_count=0
        for index,row in data.iterrows():
            byRow = data.loc[[index]].isna().sum().sum()
            if byRow > 100:
                data = data.drop(rows_count, axis=0)
            rows_count += 1

        return data   

    
    def model_learn(self):
        # Importing the dataset
        carotid_df = pd.read_csv('data/carotid_pressure.csv')
        illiac_df = pd.read_csv('data/illiac_pressure.csv')
        
        
        #Rename the first column(s)
        carotid_df.rename(columns={'Unnamed: 0':'patient_id'}, inplace=True)
        illiac_df.rename(columns={'Unnamed: 0':'patient_id'}, inplace=True)

        
        #Remove rows with more than 100 NaNs
        carotid_df = self._removeNans(carotid_df)
        illiac_df = self._removeNans(illiac_df)

        
        #Keep only the rows with indexes common to the two dataframes
        carotid_df = carotid_df.loc[carotid_df.index.isin(illiac_df.index)]
        illiac_df = illiac_df.loc[illiac_df.index.isin(carotid_df.index)]        
        
        
        # Set up X input and y target
        # Target column is the same in both csv files
        y = carotid_df[['target']].copy()
        X_carotid = carotid_df.copy()
        X_carotid = X_carotid.drop(X_carotid.columns[0],axis=1)
        X_carotid = X_carotid.drop('target', axis=1)
        X_illiac = illiac_df.copy()
        X_illiac = X_illiac.drop('target', axis=1)
        X_illiac = X_illiac.drop(X_illiac.columns[0],axis=1)
        
        
        # Clean up on sets
        X_carotid = self._cleanup(X_carotid)
        X_illiac = self._cleanup(X_illiac)
        
            
        # Splitting the dataset into the Training set and Test set.
        X_train_carotid, X_test_carotid, y_train, y_test = train_test_split(X_carotid, y, test_size = 0.20, random_state = 0)
        X_train_illiac, X_test_illiac, y_train, y_test = train_test_split(X_illiac, y, test_size = 0.20, random_state = 0)
        y_train = np.ravel(y_train)
        y_test = np.ravel(y_test)
        
        # Feature extraction? You might want to look into this to improve the model.
        # Extract and combine features
        # Feature extraction did not yield better results than removing rows with more than 100 NaNs.
        
        
        # Scale data on training set
        X_train_carotid = self.scaler_carotid.fit_transform(X_train_carotid)
        X_train_illiac = self.scaler_illiac.fit_transform(X_train_illiac)        

        # Scale data on test set
        X_test_carotid = self.scaler_carotid.fit_transform(X_test_carotid)
        X_test_illiac = self.scaler_illiac.fit_transform(X_test_illiac)
    
                
        # Combine Carotid and Illiac inputs. 
        X_train = np.hstack((X_train_carotid, X_train_illiac))
        X_test = np.hstack((X_test_carotid, X_test_illiac))
        
        
        # # Training the Naive Bayes model on the Training set
        self.classifier = RandomForestClassifier()
        self.classifier.fit(X_train, y_train)
        
        # # Predicting the Test set results
        y_pred = self.classifier.predict(X_test)

        
        # # Making the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        self.stats =  accuracy_score(y_test, y_pred)
        self.modelLearn = True

        
    def model_infer(self, time_series_carotid_filename, time_series_illiac_filename):
        if(self.modelLearn != True):
            self.model_learn()


        #Read in filenames 
        carotid_df_infer = pd.read_csv(time_series_carotid_filename)
        illiac_df_infer = pd.read_csv(time_series_illiac_filename)
        
        #Drop the first column
        carotid_df_infer = carotid_df_infer.drop(carotid_df_infer.columns[0],axis=1)
        illiac_df_infer = illiac_df_infer.drop(illiac_df_infer.columns[0],axis=1)

        # Clean up data using self._cleanup() function
        time_series_carotid = self._cleanup(carotid_df_infer)
        time_series_illiac = self._cleanup(illiac_df_infer)
        
        # Scale dataset
        time_series_carotid = self.scaler_carotid.transform(time_series_carotid)
        time_series_illiac = self.scaler_illiac.transform(time_series_illiac) 
        
        # Combine Carotid and Illiac inputs
        dataOne =  np.hstack((time_series_carotid, time_series_illiac))
        
        # Use classifier to predict the value
        y_pred = self.classifier.predict(dataOne)
        
        return y_pred

    
    def model_stats(self):
        if(self.modelLearn == False):
            self.model_learn()
        return str(self.stats)



class CarotidPressure():
    def __init__(self):
        self.modelLearn = False
        self.stats = 0
        self.scaler_carotid = StandardScaler()

        
    def _cleanup(self, data, sigma=1):
        # interpolate data
        data = data.interpolate(method='linear', axis=1)
        data = data.interpolate(method="bfill", axis=1) #Replace all remaining NaNs

        # filter noise (use scipy.ndimage.gaussian_filter1d) 
        # estimate the standard deviation (sigma)
        sigma_row = data.std(axis=1)
        if data.shape[0] > 1:      
            sigma_avg = np.mean(sigma_row)
        else:
            sigma_avg = sigma_row[0]
        
        rows=0
        for index,row in data.iterrows():
            data.iloc[rows,:] = scipy.ndimage.gaussian_filter1d(data.iloc[rows,:], sigma_avg)
            rows += 1

        return data   
    
    
    def _removeNans(self, data):
        #All rows with more than 100 NaNs are removed. 100 NaNs equals ~30% of missing data for that row. The maximum allowed per best practices.
        #Roughly 12% of all rows have 30% of missing data (more than 100 NaNs). Removing them, increased the accuracy to ~70%
        rows_count=0
        for index,row in data.iterrows():
            byRow = data.loc[[index]].isna().sum().sum()
            if byRow > 100:
                data = data.drop(rows_count, axis=0)
            rows_count += 1

        return data   

    
    def model_learn(self):
        # Importing the dataset
        carotid_df = pd.read_csv('data/carotid_pressure.csv')        
        
        #Rename the first column(s)
        carotid_df.rename(columns={'Unnamed: 0':'patient_id'}, inplace=True)
        
        #Remove rows with more than 100 NaNs
        carotid_df = self._removeNans(carotid_df)       
        
        # Set up X input and y target
        # Target column is the same in both csv files
        y = carotid_df[['target']].copy()
        X_carotid = carotid_df.copy()
        X_carotid = X_carotid.drop(X_carotid.columns[0],axis=1)
        X_carotid = X_carotid.drop('target', axis=1)      
        
        # Clean up on sets
        X_carotid = self._cleanup(X_carotid)

        # Splitting the dataset into the Training set and Test set.
        X_train_carotid, X_test_carotid, y_train, y_test = train_test_split(X_carotid, y, test_size = 0.20, random_state = 0)
        y_train = np.ravel(y_train)
        y_test = np.ravel(y_test)
        
        # Scale data on training set
        X_train_carotid = self.scaler_carotid.fit_transform(X_train_carotid) 
        
        # Scale data on test set
        X_test_carotid = self.scaler_carotid.fit_transform(X_test_carotid)   
                
        X_train = X_train_carotid
        X_test = X_test_carotid     
        
        # # Training the Naive Bayes model on the Training set
        self.classifier = RandomForestClassifier()
        self.classifier.fit(X_train, y_train)
        
        # # Predicting the Test set results
        y_pred = self.classifier.predict(X_test)

        
        # # Making the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        self.stats =  accuracy_score(y_test, y_pred)
        self.modelLearn = True
        
        
        
        
    def model_infer(self, time_series_carotid_filename):
        if(self.modelLearn != True):
            self.model_learn()

        #Read in filenames 
        carotid_df_infer = pd.read_csv(time_series_carotid_filename)
        
        #Drop the first column
        carotid_df_infer = carotid_df_infer.drop(carotid_df_infer.columns[0],axis=1)

        # Clean up data using self._cleanup() function
        time_series_carotid = self._cleanup(carotid_df_infer)
        
        # Scale dataset
        time_series_carotid = self.scaler_carotid.transform(time_series_carotid)
        
        # Combine Carotid and Illiac inputs
        dataOne =  time_series_carotid
        
        # Use classifier to predict the value
        y_pred = self.classifier.predict(dataOne)
        
        return y_pred
        
    
    def model_stats(self):
        if(self.modelLearn == False):
            self.model_learn()
        return str(self.stats)




class IlliacPressure():
    def __init__(self):
        self.modelLearn = False
        self.stats = 0
        self.scaler_illiac = StandardScaler()

        
    def _cleanup(self, data, sigma=1):
        # interpolate data
        data = data.interpolate(method='linear', axis=1)
        data = data.interpolate(method="bfill", axis=1) #Replace all remaining NaNs

        # filter noise (use scipy.ndimage.gaussian_filter1d) 
        # estimate the standard deviation (sigma)
        sigma_row = data.std(axis=1)
        if data.shape[0] > 1:      
            sigma_avg = np.mean(sigma_row)
        else:
            sigma_avg = sigma_row[0]
        
        rows=0
        for index,row in data.iterrows():
            data.iloc[rows,:] = scipy.ndimage.gaussian_filter1d(data.iloc[rows,:], sigma_avg)
            rows += 1

        return data

    def _removeNans(self, data):
        #All rows with more than 100 NaNs are removed. 100 NaNs equals ~30% of missing data for that row. The maximum allowed per best practices.
        #Roughly 12% of all rows have 30% of missing data (more than 100 NaNs). Removing them, increased the accuracy to ~70%
        rows_count=0
        for index,row in data.iterrows():
            byRow = data.loc[[index]].isna().sum().sum()
            if byRow > 100:
                data = data.drop(rows_count, axis=0)
            rows_count += 1

        return data   

    
    def model_learn(self):
        # Importing the dataset
        illiac_df = pd.read_csv('data/illiac_pressure.csv')
        
        #Rename the first column(s)
        illiac_df.rename(columns={'Unnamed: 0':'patient_id'}, inplace=True)
        
        #Remove rows with more than 100 NaNs
        illiac_df = self._removeNans(illiac_df)
        
        # Set up X input and y target
        # Target column is the same in both csv files
        y = illiac_df[['target']].copy()
        X_illiac = illiac_df.copy()
        X_illiac = X_illiac.drop('target', axis=1)
        X_illiac = X_illiac.drop(X_illiac.columns[0],axis=1)
        
        # Clean up on sets
        X_illiac = self._cleanup(X_illiac)       
            
        # Splitting the dataset into the Training set and Test set.
        X_train_illiac, X_test_illiac, y_train, y_test = train_test_split(X_illiac, y, test_size = 0.20, random_state = 0)
        y_train = np.ravel(y_train)
        y_test = np.ravel(y_test)
                  
        # Scale data on training set
        X_train_illiac = self.scaler_illiac.fit_transform(X_train_illiac)     

        # Scale data on test set
        X_test_illiac = self.scaler_illiac.fit_transform(X_test_illiac)
    
                
        # Combine Carotid and Illiac inputs. 
        X_train = X_train_illiac
        X_test = X_test_illiac       
        
        # # Training the Naive Bayes model on the Training set
        self.classifier = RandomForestClassifier()
        self.classifier.fit(X_train, y_train)
        
        # # Predicting the Test set results
        y_pred = self.classifier.predict(X_test)
        
        # # Making the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        self.stats =  accuracy_score(y_test, y_pred)
        self.modelLearn = True
        
    def model_infer(self, time_series_illiac_filename):
        if(self.modelLearn != True):
            self.model_learn()


        #Read in filenames 
        illiac_df_infer = pd.read_csv(time_series_illiac_filename)
        
        #Drop the first column
        illiac_df_infer = illiac_df_infer.drop(illiac_df_infer.columns[0],axis=1)

        # Clean up data using self._cleanup() function
        time_series_illiac = self._cleanup(illiac_df_infer)
        
        # Scale dataset
        time_series_illiac = self.scaler_illiac.transform(time_series_illiac) 
        
        # Combine Carotid and Illiac inputs
        dataOne =  time_series_illiac
        
        # Use classifier to predict the value
        y_pred = self.classifier.predict(dataOne)
        
        return y_pred
    
    def model_stats(self):
        if(self.modelLearn == False):
            self.model_learn()
        return str(self.stats)


if __name__ == '__main__':
        # m = CarotidPressure()
        # m = IlliacPressure()
        m = CombinedCardiacPressure()

        m.model_learn()

        result = m.model_infer(pd.read_csv('data/carotid_pressure_test_1.csv'), pd.read_csv('data/illiac_pressure_test_1.csv'))
        print(result)

        result = m.model_infer(pd.read_csv('data/carotid_pressure_test_2.csv'), pd.read_csv('data/illiac_pressure_test_2.csv'))
        print(result)

        result = m.model_infer(pd.read_csv('data/carotid_pressure_test_3.csv'), pd.read_csv('data/illiac_pressure_test_3.csv'))
        print(result)

        result = m.model_infer(pd.read_csv('data/carotid_pressure_test_4.csv'), pd.read_csv('data/illiac_pressure_test_4.csv'))
        print(result)
            
        print(m.model_stats())

