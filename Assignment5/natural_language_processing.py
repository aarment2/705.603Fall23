# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


class Sentiment():
    def __init__(self):
        self.modelLearn = False
        self.stats = 0
        self.stopwords = False
        self.vectorizer = CountVectorizer()
        self.test = 0
        
    def _cleanup(self, data, lines):
        
        corpus = []
        
        if(self.stopwords == False):
            #download the stop words
            nltk.download('stopwords')
            self.stopwords_set = set(stopwords.words('english'))
            self.stopwords = True
        
        ps = PorterStemmer()
        
        for i in range(0, lines):
          # Get words only using regex in re
            review = re.sub('[^a-zA-Z]', ' ', data['Review'][i])
            
          # Make all lower case
            review = review.lower()
          
          # split into an array of strings necessary to remove stop words
            review = review.split()
          
          # Collect Stop words
            all_stopwords = self.stopwords_set
        
          # Add back in words that may influence the sentiment such as 'not'
            all_stopwords.discard("aren't")
            all_stopwords.discard("most")
            all_stopwords.discard("more")
            all_stopwords.discard("very")
            all_stopwords.discard("isn't")
            all_stopwords.discard("don't")
            all_stopwords.discard("above")
            all_stopwords.discard("below")
            all_stopwords.discard("not")
        
          # Stem the words and filter out stopwords
            review = [ps.stem(word) for word in review if not word in all_stopwords]
         
          # Turn back into a string
            review = ' '.join(review)
            
          # Append the string to the total
            corpus.append(review)
        return corpus

    def model_learn(self):
        # Importing the dataset
        dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

        corpus = self._cleanup(dataset, len(dataset))
 
        # Creating the Bag of Words model

        # fit to an array using fit_transform
        X = self.vectorizer.fit_transform(corpus).toarray()
                
        # Set the label (1 for good, 0 for bad)
        y = dataset['Liked'].values

        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

        # Training the Naive Bayes model on the Training set
        from sklearn.naive_bayes import GaussianNB
        self.classifier = GaussianNB()
        self.classifier.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = self.classifier.predict(X_test)

        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix, accuracy_score
        cm = confusion_matrix(y_test, y_pred)
        
        self.stats =  accuracy_score(y_test, y_pred)
        self.modelLearn = True
        
    def model_infer(self, captureString):
        if(self.modelLearn != True):
            self.model_learn()
            
        # Build 1 entry dictionary similar to Reviews structure with Review:String
        entry = {'Review': [captureString]}
        
        # Convert into a dataframe
        dataOne = pd.DataFrame(entry)
        
        # Cleanup the dataframe
        oneline = self._cleanup(dataOne,1)
        
        # Transform the datafame to an array using transform
        XOne = self.vectorizer.transform(oneline).toarray()
        
        # Use classifier to predict the value
        y_pred = self.classifier.predict(XOne)
        self.test = y_pred
        
        return y_pred > 0
    
    def model_stats(self):
        if(self.modelLearn == False):
            self.model_learn()
        return str(self.stats)

#if __name__ == '__main__':
#        m = Sentiment()
#        m.model_learn()
#        result = m.model_infer("bad terrible stinks horrible")
#        if( result > 0):
#            print("Good")
#        else:
#            print("Bad")
#        result = m.model_infer("fantastic wonderful super good")
#        if( result > 0):
#            print("Good")
#        else:
#            print("Bad")
#            
#        print( m.model_stats())
