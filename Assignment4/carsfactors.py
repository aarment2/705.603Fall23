# Multiple Linear Regression

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class carsfactors:
    def __init__(self):
        self.modelLearn = False
        self.stats = 0
        self.enc_bodytype = None
        self.enc_transmission = None
        self.enc_color = None
        self.scaler = None
        self.regressor = None

    def model_learn(self):
        # Importing the dataset into a pandas dataframe
        df=pd.read_csv('cars.csv')

        
        #Remove Unwanted Columns - 'manufacturer_name', 'model_name', 'engine_fuel','engine_has_gas', 'engine_type', 'engine_capacity','has_warranty', 'is_exchangeable', 'state', 'location_region', drivetrain',  'number_of_photos','up_counter', 'feature_0', 'feature_1','feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7'
        df = df.drop(columns=['manufacturer_name', 'model_name', 'engine_fuel','engine_has_gas', 'engine_type', 'engine_capacity','has_warranty', 'is_exchangeable', 'state', 'location_region', 'drivetrain',  'number_of_photos','up_counter', 'feature_0', 'feature_1','feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7','feature_8','feature_9'])
 

        # Seperate X and y (features and label)  The last feature "duration_listed" is the label (y)
        # Seperate X vs Y
        X = df[['transmission','color','odometer_value','year_produced','body_type','price_usd']].copy()
        y = df[['duration_listed']].copy()
 

        # Do the ordinal Encoder for car type to reflect that some cars are bigger than others.  
        # This is the order 'universal','hatchback', 'cabriolet','coupe','sedan','liftback', 'suv', 'minivan', 'van','pickup', 'minibus','limousine'
        # make sure this is the entire set by using unique()
        # create a seperate dataframe for the ordinal number - so you must strip it out and save the column
        # make sure to save the OrdinalEncoder for future encoding due to inference
        body_types_unique = np.unique(df[["body_type"]])
        #print(body_types_unique)        
        enc_bodytype = OrdinalEncoder(categories=[['universal','hatchback','cabriolet','coupe','sedan','liftback','suv','minivan','van','pickup','minibus','limousine']])
        bodytype_enc = enc_bodytype.fit_transform(df[["body_type"]])
        bodytype_enc_df = pd.DataFrame(bodytype_enc, columns = ['body_type(0)'])
        self.enc_bodytype = enc_bodytype
        

        # Do onehotencoder for Transmission only - again you need to make a new dataframe with just the encoding of the transmission
        # save the OneHotEncoder to use for future encoding of transmission due to inference
        enc_transmission = OneHotEncoder(sparse_output=False)
        transmission_enc = enc_transmission.fit_transform(df[["transmission"]])
        transmission_enc_df = pd.DataFrame(transmission_enc, columns = ['transmission(0)','transmission(1)'])
        self.enc_transmission = enc_transmission
        
        
        # Do onehotencoder for Color
        # Save the OneHotEncoder to use for future encoding of color for inference
        enc_color = OneHotEncoder(sparse_output=False)
        color_enc = enc_color.fit_transform(df[["color"]])
        color_enc_df = pd.DataFrame(color_enc, columns = ['color(0)','color(1)','color(2)','color(3)','color(4)','color(5)','color(6)','color(7)','color(8)','color(9)','color(10)','color(11)'])
        self.enc_color = enc_color
        

        # the all three together endocdings into 1 data frame (need 2 steps with "concatenate")
        # add the ordinal and transmission then add color
        concat_df = pd.concat([transmission_enc_df,color_enc_df, bodytype_enc_df], axis=1)
        

        # then dd to original data set
        X = pd.concat([X,concat_df], axis=1)
        
                
        #delete the columns that are substituted by ordinal and onehot - delete the text columns for color, transmission, and car type
        X = X.drop(columns=['transmission', 'color', 'body_type'])         

        
        # Splitting the dataset into the Training set and Test set - use trian_test_split
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, shuffle=True)
                 
                
        # Feature Scaling - required due to different orders of magnitude across the features
        # make sure to save the scaler for future use in inference
        scaler = StandardScaler()
        self.scaler = scaler
        scaled_columns = scaler.fit_transform(X_train[['odometer_value','year_produced','price_usd']])
        X_train[['odometer_value','year_produced','price_usd']] = scaled_columns
        scaled_columns = scaler.fit_transform(X_test[['odometer_value','year_produced','price_usd']])
        X_test[['odometer_value','year_produced','price_usd']] = scaled_columns

        
        # Training the Multiple Linear Regression model on the Training set
        from sklearn.linear_model import LinearRegression
        self.regressor = LinearRegression()
        self.regressor.fit(X_train, y_train)
        
        self.stats = self.regressor.score(X_train, y_train)
        self.modelLearn = True

        
    # this demonstrates how you have to conver these values using the encoders and scalers above
    def model_infer(self,transmission, color, odometer, year, bodytype, price):
        if(self.modelLearn == False):
            self.model_learn()

        #convert the body type into a numpy array that holds the correct encoding
        carTypeTest = self.enc_bodytype.transform([[bodytype]])
        
        #convert the transmission into a numpy array with the correct encoding
        carHotTransmissionTest = self.enc_transmission.transform([[transmission]])
        
        #conver the color into a numpy array with the correct encoding
        carHotColorTest = self.enc_color.transform([[color]])
        
        #add the three above
        total = np.concatenate((carHotTransmissionTest,carHotColorTest), 1)
        total = np.concatenate((total,carTypeTest), 1)
        
        # build a complete test array and then predict
        othercolumns = np.array([[odometer ,year, price]])
        othercolumns = self.scaler.transform(othercolumns)  
        totaltotal = np.concatenate((othercolumns,total),1)

        #must scale
#        attempt = self.scaler.transform(odometer, year, price)  
        
        #determine prediction
        y_pred = self.regressor.predict(totaltotal)
        return str(y_pred)
        
    def model_stats(self):
        if(self.modelLearn == False):
            self.model_learn()
        return str(self.stats)
