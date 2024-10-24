import streamlit as st
from codes.col1_datacleaning import col1_age, col1_fare, col1_cabin, col1_embarked
from codes.col2_datacleaning import col2_age, col2_fare, col2_cabin, col2_embarked, col2_relatives, col2_title, col2_numeric_features
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

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Cleaning", "Predictie","Visualisaties Orgineel","Visualisaties Verbeterd", "bronnen"])



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
    
    col1, col2 = st.columns([1, 1])
    with col2:
        col2_numeric_features(st)

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

with tab5:
    text = """
    ### Bronnen voor het maken van plots en Machine Learning

    Het maken van de plotten met behulp van Plotly is voornamelijk gedaan met de kennis uit **DataCamp**. 
    - [DataCamp](https://app.datacamp.com/learn/assignments)

    Voor de Machine Learning hebben we gebruik gemaakt van dit voorbeeld op **Kaggle**:
    - [Kaggle Titanic Tutorial](https://www.kaggle.com/code/alexisbcook/titanic-tutorial)

    Voor het oplossen van errors die we kregen in de code en voor hulp bij het maken van plots:
    - [ChatGPT](https://chatgpt.com/)

    Dit is een voorbeeld dat we vonden over hoe iemand anders de Titanic-opdracht heeft uitgevoerd, maar we hebben dit achteraf niet echt gebruikt:
    - [GitHub - Titanic Classification](https://github.com/murilogustineli/Titanic-Classification/blob/main/Titanic%20Project.ipynb)
    """
    st.markdown(text)