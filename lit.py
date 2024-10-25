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

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Data Cleaning", "Predictie", "Visualisaties Orgineel", "Visualisaties Verbeterd","Map", "Bronnen"])

# Function definitions remain the same
def show_data_cleaning():
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

def show_prediction():
    col1, col2 = st.columns([1, 1])
    with col1:
        st.header('Origineel')
        st.write('---')
    with col2:
        st.header('Verbeterd')
        st.write('---')
    prediction(st)

def show_original_plots():
    test_data = pd.read_csv('test.csv')
    train_data = pd.read_csv('train.csv')
    og_plots(st, pd, train_data, test_data)

def show_new_plots():
    col1, col2 = st.columns([1, 1])
    with col1:
        st.header('Train Data')
        st.write('---')
    with col2:
        st.header('Test Data')
        st.write('---')
    
    new_test_data = pd.read_csv('test_categorical.csv')
    new_train_data = pd.read_csv('train_categorical.csv')
    new_plots(st, pd, np, new_train_data, new_test_data)


def show_map():
    import plotly.graph_objects as go
    import streamlit as st

    df = pd.read_csv('train_categorical.csv')

    # Titanic's route coordinates (in latitude, longitude format)
    locations = {
        "Southampton, UK": (50.8998, -1.4044),
        "Cherbourg, France": (49.634, -1.622),
        "Queenstown, Ireland": (51.849, -8.294),
        "Final Position": (41.7325, -49.9469)  # Approximate sinking coordinates
    }

    #df = pd.read_csv('train_categorical.csv')

    embarked_counts = {
    "Southampton, UK": {
        "total": df[df['Embarked'] == 'S'].shape[0],
        "class_1": df[(df['Embarked'] == 'S') & (df['Pclass'] == 1)].shape[0],
        "class_2": df[(df['Embarked'] == 'S') & (df['Pclass'] == 2)].shape[0],
        "class_3": df[(df['Embarked'] == 'S') & (df['Pclass'] == 3)].shape[0],
    },
    "Cherbourg, France": {
        "total": df[df['Embarked'] == 'C'].shape[0],
        "class_1": df[(df['Embarked'] == 'C') & (df['Pclass'] == 1)].shape[0],
        "class_2": df[(df['Embarked'] == 'C') & (df['Pclass'] == 2)].shape[0],
        "class_3": df[(df['Embarked'] == 'C') & (df['Pclass'] == 3)].shape[0],
    },
    "Queenstown, Ireland": {
        "total": df[df['Embarked'] == 'Q'].shape[0],
        "class_1": df[(df['Embarked'] == 'Q') & (df['Pclass'] == 1)].shape[0],
        "class_2": df[(df['Embarked'] == 'Q') & (df['Pclass'] == 2)].shape[0],
        "class_3": df[(df['Embarked'] == 'Q') & (df['Pclass'] == 3)].shape[0],
    },
}

    # Extract coordinates for the route
    latitudes = [coord[0] for coord in locations.values()]
    longitudes = [coord[1] for coord in locations.values()]
    place_names = list(embarked_counts.keys())

    hover_texts = [
    f"{place}<br>"
    f"Totaal ingestapte Passagiers: {embarked_counts[place]['total']}<br>"
    f"Klasse 1: {embarked_counts[place]['class_1']}<br>"
    f"Klasse 2: {embarked_counts[place]['class_2']}<br>"
    f"Klasse 3: {embarked_counts[place]['class_3']}" for place in embarked_counts.keys()
]

    # Calculate the mean latitude and longitude for centering
    mean_lat = sum(latitudes) / len(latitudes)
    mean_lon = sum(longitudes) / len(longitudes)

    # Create the map figure
    fig = go.Figure()

    # Add the route as a scattergeo line plot
    fig.add_trace(go.Scattergeo(
        locationmode = 'ISO-3',
        lat = latitudes,
        lon = longitudes,
        mode = 'lines+markers',
        text = hover_texts, 
        textposition = "top center",
        textfont = dict(color='black', size=12, weight="bold" ),
        line = dict(width = 2, color = 'orchid'),
        marker = dict(size = 7, color = 'purple'),
    ))

    # Customize map layout
    fig.update_layout(
        title = {
            'text': 'Titanic Route Map',
            'font': {'size': 24},
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        },
        geo = dict(
            projection_type = 'mercator',
            showcountries = True,
            coastlinecolor = "black",
            showland = True,
            landcolor = "rgb(229, 236, 246)",
            countrycolor = "black",
            showocean = True,
            oceancolor = "lightblue",
            center = dict(
                lon = mean_lon,  # Center at mean longitude
                lat = mean_lat   # Center at mean latitude
            ),
            lonaxis = dict(
                range = [mean_lon - 40, mean_lon + 40]  # Increased Longitude range to zoom out
            ),
            lataxis = dict(
                range = [mean_lat - 20, mean_lat + 20]  # Increased Latitude range to zoom out
            ),
        ),
        width = 1000,    # Increased width
        height = 800,    # Increased height
    )
    

    # Display the map in Streamlit
    st.plotly_chart(fig)


def show_sources():
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

# Load content in tabs
with tab1:
    show_data_cleaning()
with tab2:
    show_prediction()
with tab3:
    show_original_plots()
with tab4:
    show_new_plots()
with tab5:
    show_map()
with tab6:
    show_sources()