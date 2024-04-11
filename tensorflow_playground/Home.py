# imporiting Libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import keras



st.header('Tensorflow Playground')
st.divider()
st.image('data/Neural_Network.png')
st.divider()
with st.expander("Playground - 1"):
    st.markdown('''In :red[Playground 1]    
                You can Create :blue[Random] Data Set and Perform Experiments    
                :red[Remember:] Whenever you click on :blue[Playground 1] You may see the :red[Error!], 
                Enter :red[n_samples] it will disappear automatically.''')
with st.expander("Playground - 2"):
    st.markdown('''In :red[Playground 2]    
                You can Select the :blue[Preloaded] Data Set and Perform Experiments''')