import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import LabelEncoder
import scipy.stats as ss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer,FunctionTransformer,LabelEncoder,OneHotEncoder,OrdinalEncoder,StandardScaler
from mixed_naive_bayes import MixedNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score,classification_report
import random
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline,make_pipeline
import pickle
from sklearn.impute import SimpleImputer
import streamlit as st

preprocesser = pickle.load(open(r"E:\projects\Drug_feature_extraction.pkl",'rb'))

model = pickle.load(open(r"E:\projects\Drug_model.pkl",'rb'))

st.title('Classification of Drugs')

age = st.slider('Enter age',min_value= 1,max_value=100)
sex = st.radio('Select Gender',['Male','Female'])
bp = st.radio('Select BP',['HIGH','LOW','NORMAL'])
cholesterol = st.radio('Select cholesterol',['HIGH','NORMAL'])
na_to_k = st.slider('Enter Na_to_K level',min_value= 0.0,max_value= 40.0, step= 0.001)

if sex == 'Male':
    sex = 'M'
else:
    sex = 'F'

query = pd.DataFrame([[age,sex,bp,cholesterol,na_to_k]],
                     columns= ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'])

pred = preprocesser.transform(query)

res = model.predict(pred)

if res == 0:
    res = "DrugY"
elif res == 1:
    res = "drugA"
elif res == 2:
    res = "drugB"
elif res == 3:
    res = "drugC"
else:
    res = "drugX"

if st.button('Submit'):
    st.header(res)
