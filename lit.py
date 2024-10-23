import streamlit as st
from codes.col1_datacleaning import col1_age, col1_fare, col1_cabin, col1_embarked
from codes.col2_datacleaning import col2_age, col2_fare, col2_cabin, col2_embarked, col2_relatives, col2_title
from codes.og_plots import og_plots
from codes.new_plots import new_plots
from codes.prediction import prediction

import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Titanic Case",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title('Titanic Case')

tab1, tab2, tab3, tab4, = st.tabs(["Data Cleaning", "Predictie","Visualisaties Orgineel","Visualisaties Verbeterd"])



with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header('Origineel')
        st.write('---')
        col1_age(st)
    with col2:
        st.header('Verbeterd')
        st.write('---')
        col2_age(st)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        col1_fare(st)
    with col2:
        col2_fare(st)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        col1_cabin(st)
    with col2:
        col2_cabin(st)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        col1_embarked(st)
    with col2:
        col2_embarked(st)
    
    col1, col2 = st.columns([1, 1])
    with col2:
        col2_relatives(st)
    
    col1, col2 = st.columns([1, 1])
    with col2:
        col2_title(st)

with tab2:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header('Origineel')
        st.write('---')
    
    with col2:
        st.header('Verbeterd')
        st.write('---')
    
    prediction(st)

with tab3:
    test_data = pd.read_csv('test.csv')
    train_data = pd.read_csv('train.csv')
    og_plots(st, pd, train_data, test_data)

with tab4:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header('Train Data')
        st.write('---')

    with col2:
        st.header('Test Data')
        st.write('---')

    new_test_data = pd.read_csv('test_categorical.csv')
    new_train_data = pd.read_csv('train_categorical.csv')
    new_plots(st, pd , np, new_train_data, new_test_data)

