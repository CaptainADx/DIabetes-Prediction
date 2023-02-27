# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 02:08:52 2023

@author: capta
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler


#Reading the csv file.
diabetes_dataset = pd.read_csv('C:/Users/capta/Data Science/Major Project/diabetes.csv')

#RENAMING A COLUMN
diabetes_dataset.rename(columns = {'Outcome' : 'outcome'},inplace = True)


#SEPARATING THE COLUMNS
x = diabetes_dataset.drop(columns = ['outcome'], axis = 1)
y = diabetes_dataset['outcome']


#DATA STANDARDIZATION
dataframe = diabetes_dataset
x = dataframe[dataframe.columns[:-1]].values
y = dataframe[dataframe.columns[-1]].values
scaler = StandardScaler()
scaler.fit(x)
standardize_data = scaler.transform(x)
x = standardize_data


#SPLITTING THE TRAIN AND TEST DATA
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, stratify = y , random_state = 2)


#TRAINING THE DATA
classifier = svm.SVC(kernel = 'linear')
classifier.fit(x_test, y_test)


#LOADING THE SAVED MODEL
loaded = pickle.load(open('C:/Users/capta/Data Science/Major Project/trained_model.sav', 'rb'))

#Inputting A Raw Data
input_data = (5,121,72,23,112,26.2,0.245,30)

#Changing The Input Data To Numpy Array
input_data_as_numpy_array = np.asarray(input_data)

#Reshaping The Data As We Are Predicting The Outcome For Only One Input
reshaped_data = input_data_as_numpy_array.reshape(1,-1)

#Standardizing The Input Data
standard_data = scaler.transform(reshaped_data)
input_data = standard_data
# print(input_data)

#Predicting The Outcome
prediction = loaded.predict(input_data)
# print(prediction)

#Outputting The Result In Readable Format
if (prediction == 0):
    print('The Patient is Non-Diabetic')
elif(prediction == 1):
    print('The Patient is Diabetic')
else:
    print('Error !')
    
    





