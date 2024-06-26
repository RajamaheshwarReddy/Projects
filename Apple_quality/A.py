# -*- coding: utf-8 -*-
"""classify Apple_quality_(pipeline).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1v7fo9wPRWmD1aopWfU8hoM3nwQP14cRX

## Problem Statement
- Create a model to classify the Quality of an apple based on their features

##1. Importing libraries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline,make_pipeline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.naive_bayes import GaussianNB,MultinomialNB

import streamlit as s

#"""## 2. Data Collection"""

# collect the dataset from kaggle,which is used to create a good model for classification
data = pd.read_csv(r"E:\projects\apple_quality.csv")
#data.head()

#data.iloc[3]

#"""## 3. EDA and 4.Pre-processing"""

#data.shape   # data set contains 4001 rows and 9 columns

#data.info()

#"""- This data contains some numerical data and categorical data.
#- Column ACIDITY  is a numerical column so we can convert obejct to numerical type.
#- This data contains one missing row.So we can simply remove those row

#"""

data.drop(4000,axis = 0,inplace = True)   # remove missing row,which contain last row in the data set.

data["Acidity"] = data["Acidity"].astype("float")   # we can convert Acidity type from object to numerical

#data.info()

#data.duplicated().sum()    # there is no duplicated values are here.

#data.Quality.value_counts()  # balanced data



# we don't have any use of A_id column ,so simply remove those column
fv = data.iloc[:,1:-1]
cv = data.iloc[:,-1]
#fv

# convert good as 0 and bad as 1
cv = cv.map({"good":0,"bad":1})

#cv

#"""## 5.EDA after Pre-processing"""

# Our features are contains numerical data type.

#"""Based on above graph and table, all the features are indipendent only.So we will go with Naive Bayes theroem
#- our dataset is a numerical data set so we will go with Gaussian distribution.
#- then check each feature follows gaussian distribution or not

##### By using Q-Q plot,I find that those columns are follows Gaussian distribution or Not
#"""



#"""Based on above graphs, all the features follows Gaussian distribution,So we performe Gaussian Naive bayes"""



#"""## 6.Feature Engineering"""

# split feature variables and class variables as x_train amd x_test
x_train,x_test,y_train,y_test = train_test_split(fv,cv,test_size=0.2,stratify=cv)

#x_train.info()

# split x_train into category wise
#my data contains numerical only,so split my data as numerical_data

numerical_data = x_train.select_dtypes(include = ["int64","float64"])

num_pip = Pipeline([("imputer",SimpleImputer()) ,("Standardization",StandardScaler())])

#numerical_data.columns

ctp = ColumnTransformer([("numerical",num_pip,numerical_data.columns)],remainder = "passthrough")

#pd.DataFrame(ctp.fit_transform(x_train),columns=x_train.columns)

final_pre=Pipeline([("Pre-Processing",ctp)])

import pickle
pickle.dump(final_pre,open(r"E:\projects\final_pre.pkl","wb"))



#"""## 7.Training"""

# x_train follows gaussian distribution so i can use GaussianNB

# create an object of GaussianNB class
g = GaussianNB()

# fit x_train and y_train based on gaussian NB
model = g.fit(final_pre.fit_transform(x_train),y_train)

import pickle
pickle.dump(model,open(r"E:\projects\fmodel.pkl","wb"))

#"""then finally we create a model for the Gaussian Naive bayes

## 8. Model Evalution
#"""

from sklearn.metrics import confusion_matrix,classification_report

confusion_matrix(y_test,model.predict(final_pre.transform(x_test)))

print(classification_report(y_test,model.predict(final_pre.transform(x_test))))





pre=pickle.load(open(r"E:\projects\final_pre.pkl","rb"))
model=pickle.load(open(r"E:\projects\fmodel.pkl","rb"))

#model=pickle.load(open(r"C:\Users\lenovo\Downloads\final_model.pkl","rb"))
#s.title("Predicting Titanic Survival")
s.title("Apple Quality Prediction web app")
size=s.number_input("type size")
weight=s.number_input("type weight")
sweetness=s.number_input("type sweetness")
crunchiness=s.number_input("type crunchiness")
juice=s.number_input("type juice")
ripe=s.number_input("type ripe")
acidity=s.number_input("type acidity")

query = pd.DataFrame([[size,weight,sweetness,crunchiness,juice,ripe,acidity]],
                  columns= ['size','weight','sweetness','crunchiness','juice','ripe','acidity'])


q=pre.transform(query)
predi=model.predict(q)

#predi=model.predict([[size,weight,sweetness,crunchiness,juice,ripe,acidity]])

if predi==1:
    x="good"
else:
    x="bad"
    


if s.button('submit'):
    s.write('"your apple quality is'+x)

