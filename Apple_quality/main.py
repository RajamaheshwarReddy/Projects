import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 
import sklearn
import pickle
import streamlit as s


#pre=pickle.load(open(r"/content/drive/MyDrive/pre.pkl","rb"))
#model=pickle.load(open(r"/content/drive/MyDrive/pre.pkl","rb"))

model=pickle.load(open(r"C:\Users\lenovo\Downloads\final_model.pkl","rb"))
#s.title("Predicting Titanic Survival")
s.title("Apple Quality Prediction")
size=s.number_input("type size")
weight=s.number_input("type weight")
sweetness=s.number_input("type sweetness")
crunchiness=s.number_input("type crunchiness")
juice=s.number_input("type juice")
ripe=s.number_input("type ripe")
acidity=s.number_input("type acidity")

# q=pre.transform([[size,weight,sweetness,crunchiness,juice,ripe,acidity]])
# predi=model.predict(q)

predi=model.predict([[size,weight,sweetness,crunchiness,juice,ripe,acidity]])

#'''
#print("hi")

#s.write("hello welcome to streamlit")
#s.write("bye")
#s.radio("what is choice",["bad","good"])
#var =s.text_input("type ur name")
#print(var)
#num=s.number_input("type number")
#print(num)
#'''

#'''if predi==1:
#    s.write("your apple quality is good")
#else:
#    s.write("your apple quality is bad")
#'''
if predi==1:
    x="good"
else:
    x="bad"
    


if s.button('submit'):
    s.write('"your apple quality is'+x)

#df=pd.read_csv(r"C:\Users\lenovo\Downloads\apple_quality.csv")
#s.write(df)
#s.write("your apple quality is {}".format(predi))
#s.scatter_chart(df,x="Size",y="Weight")