import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def og_plots(st,pd, df, df_test):
    col1, col2 = st.columns([1, 1])

    with col1:

        # Verwijder de kolommen die niet numeriek zijn
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        m = df[numeric_columns].corr()

        fig, ax = plt.subplots(figsize=(7,5))
        sns.heatmap(m, annot=True, cmap="Reds", ax=ax)
        
        # Display the heatmap using Streamlit
        st.subheader('Correlatie matrix - train data')
        st.pyplot(fig, clear_figure=True)
    
    with col2:
        # Verwijder de kolommen die niet numeriek zijn
        numeric_columns = df_test.select_dtypes(include=[np.number]).columns
        m = df_test[numeric_columns].corr()

        fig, ax = plt.subplots(figsize=(7,5))
        sns.heatmap(m, annot=True, cmap="Reds", ax=ax)
        
        # Display the heatmap using Streamlit
        st.subheader('Correlatie matrix - test data')
        st.pyplot(fig, clear_figure=True)

    col1, col2 = st.columns([1, 1])
    with col1:
                # Create two new dataframes for survived and not survived passengers
        df_survived = df[df['Survived'] == 1]
        df_not_survived = df[df['Survived'] == 0]

        # Set up the figure for the histograms
        fig, ax = plt.subplots()

        # Create histograms
        ax.hist(df_survived['Age'].dropna(), bins=15, alpha=0.5, label='Survived')
        ax.hist(df_not_survived['Age'].dropna(), bins=15, alpha=0.5, label='Not Survived')

        # Add labels and title
        ax.set_xlabel('Age')
        ax.set_ylabel('Number of Passengers')
        ax.set_title('Age Distribution of Passengers')

        # Add a legend
        ax.legend()

        # Display the plot using Streamlit
        st.subheader('Leeftijd verdeling van passagiers')
        st.pyplot(fig)

    with col2:
        # er zijn nog steeds missende waarden in de Age kolom
        # we kunnen deze vervangen door de gemiddelde leeftijd van de passagiers te nemen
        mean_age = df['Age'].mean()
        df['Age'] = df['Age'].fillna(mean_age)

        mean_age = df_test['Age'].mean()
        df_test['Age'] = df_test['Age'].fillna(mean_age)

        # groepeer de leeftijden in bins van 10 jaar
        bins = [0, 18, 30, 40, 50, 60, 70, 80]
        df['Age_group'] = pd.cut(df['Age'], bins)
        df_test['Age_group'] = pd.cut(df_test['Age'], bins)

        # plot de overlevingskans per leeftijdsgroep
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(x='Age_group', hue='Survived', data=df, ax=ax)
        ax.set_title('Survival Rate per Age Group')
        st.subheader('Overlevingskans per leeftijdsgroep')
        st.pyplot(fig)

    col1, col2 = st.columns([1, 1])

    with col1:
        # maak subplot een scatter plot van de ticket prijs tegen de overlevenden en niet overlevenden
        fig, ax = plt.subplots()
        ax.scatter(df_survived['Fare'], df_survived['Age'], label='Survived')
        ax.scatter(df_not_survived['Fare'], df_not_survived['Age'], label='Not survived')
        ax.set_xlabel('Fare')
        ax.set_ylabel('Age')
        ax.set_title('Fare against Survival')
        ax.legend()

        st.subheader("Ticket prijs tegen overlevenden en niet overlevenden")
        st.pyplot(fig)
    with col2:
        # maak een bar plot van de gemiddelde ticket prijs voor overlevenden en niet overlevenden
        fig, ax = plt.subplots()
        ax.bar('Survived', df_survived['Fare'].mean(), color='lightgreen', label='Survived')
        ax.bar('Not survived', df_not_survived['Fare'].mean(), color='coral', label='Not survived')
        ax.set_ylabel('Fare')
        ax.set_title('Mean fare for survived and not survived')
        ax.legend()

        st.subheader("Overlevingskans per gemiddelde ticket prijs")
        st.pyplot(fig)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Set up the figure for the boxplot
        fig, ax = plt.subplots(figsize=(7, 5))

        # Create boxplot comparing ticket prices by survival status
        sns.boxplot(x='Survived', y='Fare', data=df, ax=ax)

        # Replace x-axis labels with 'Not survived' and 'Survived'
        ax.set_xticklabels(['Not survived', 'Survived'])

        # Add a title
        ax.set_title('Ticket Price Distribution for Survived vs. Not Survived')

        # Display the plot using Streamlit\
        st.subheader('Boxplot van ticket prijzen per overlevingsstatus')
        st.pyplot(fig)
    
    with col2:
        # Kolom 'Fare' heeft 1 onbekende waarde in de test data, dus we vervangen deze door de gemiddelde waarde
        mean_fare = df_test['Fare'].mean()
        df_test['Fare'] = df_test['Fare'].fillna(mean_fare)
        # groepeer de ticket prijzen in bins. de max is 513
        bins = [0, 5, 10, 15, 20, 25, 50, 100, 200, 300, 513]
        df['Fare_group'] = pd.cut(df['Fare'], bins)
        df_test['Fare_group'] = pd.cut(df_test['Fare'], bins)
        df['Fare_group'] = df['Fare_group'].fillna(df['Fare_group'].mode()[0])
        df_test['Fare_group'] = df_test['Fare_group'].fillna(df_test['Fare_group'].mode()[0])

        # Set up the figure for the count plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create count plot for survival rate per passenger class
        sns.countplot(x='Pclass', hue='Survived', data=df, ax=ax)

        # Add a title to the plot
        ax.set_title('Survival Rate per Passenger Class')

        # Display the plot using Streamlit
        st.subheader('Overlevingskans per passagiersklasse')
        st.pyplot(fig)

    col1, col2 = st.columns([1, 1])

    with col1:
        # Group by 'SibSp' and 'Survived' and calculate proportions
        df_sibsp = df.groupby(['SibSp', 'Survived']).size().unstack()
        df_sibsp_normalized = df_sibsp.div(df_sibsp.sum(axis=1), axis=0)

        # Set up the figure for the bar plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create a stacked bar plot
        df_sibsp_normalized.plot(kind='bar', stacked=True, ax=ax)

        # Add labels and title
        ax.set_ylabel('Proportion of Passengers')
        ax.set_title('Survival Proportion per SibSp')

        # Display the plot using Streamlit
        st.subheader('Overlevingskans per SibSp')
        st.pyplot(fig)
    
    with col2:
        # Set up the figure for the count plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create count plot for survival per SibSp
        sns.countplot(x='SibSp', hue='Survived', data=df, ax=ax)

        # Add a title to the plot
        ax.set_title('Survival per SibSp')

        # Display the plot using Streamlit
        st.subheader('Overlevingskans per SibSp')
        st.pyplot(fig)
    
    col1, col2 = st.columns([1, 1])

    with col1:
        # Set up the figure for the count plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create count plot for age group per SibSp
        sns.countplot(x='SibSp', hue='Age_group', data=df, ax=ax)

        # Add a title to the plot
        ax.set_title('Age Group Distribution per SibSp')

        # Display the plot using Streamlit
        st.subheader('Leeftijdsgroep verdeling per SibSp')
        st.pyplot(fig)

    with col2:
        # Group by 'Parch' and 'Survived' and calculate proportions
        df_parch = df.groupby(['Parch', 'Survived']).size().unstack()
        df_parch_normalized = df_parch.div(df_parch.sum(axis=1), axis=0)

        # Set up the figure for the bar plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create a stacked bar plot
        df_parch_normalized.plot(kind='bar', stacked=True, ax=ax)

        # Add labels and title
        ax.set_ylabel('Proportion of Passengers')
        ax.set_title('Survival Proportion per Parch')

        # Display the plot using Streamlit
        st.subheader('Overlevingskans per Parch')
        st.pyplot(fig)
    
    col1, col2 = st.columns([1, 1])

    with col1:
        # Set up the figure for the count plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create count plot for age group per Parch
        sns.countplot(x='Parch', hue='Age_group', data=df, ax=ax)

        # Add a title to the plot
        ax.set_title('Age Group Distribution per Parch')

        # Display the plot using Streamlit
        st.subheader('Leeftijdsgroep verdeling per Parch')
        st.pyplot(fig)
    
    with col2:
        # Group by 'Embarked' and 'Survived' and calculate proportions
        df_embarked = df.groupby(['Embarked', 'Survived']).size().unstack()
        embarked_normalized = df_embarked.div(df_embarked.sum(axis=1), axis=0)  # Normalize by row

        # Set up the figure for the bar plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create a stacked bar plot
        embarked_normalized.plot(kind='bar', stacked=True, ax=ax)

        # Add labels and title
        ax.set_ylabel('Proportion of Passengers')
        ax.set_title('Survival Proportion per Embarkation Point')

        # Display the plot using Streamlit
        st.subheader('Overlevingskans per opstap punt')
        st.pyplot(fig)

    col1, col2 = st.columns([1, 1])

    with col1:
        # Calculate the mean fare per embarkation point
        mean_fare = df.groupby('Embarked')['Fare'].mean()

        # Set up the figure for the bar plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create a bar plot for the mean fare
        mean_fare.plot(kind='bar', ax=ax)

        # Add labels and title
        ax.set_ylabel('Mean Fare')
        ax.set_title('Mean Fare per Embarkation Point')

        # Display the plot using Streamlit
        st.subheader('Gemiddelde ticket prijs per opstap punt')
        st.pyplot(fig)
    
    with col2:
        # Calculate the mean fare per embarkation point
        mean_fare = df.groupby('Embarked')['Age'].mean()

        # Set up the figure for the bar plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create a bar plot for the mean fare
        mean_fare.plot(kind='bar', ax=ax)

        # Add labels and title
        ax.set_ylabel('Mean Age')
        ax.set_title('Mean Age per Embarkation Point')

        # Display the plot using Streamlit
        st.subheader('Gemiddelde leeftijd per opstap punt')
        st.pyplot(fig)
    
    col1, col2 = st.columns([1, 1])

    with col1:
        # Set up the figure for the count plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create count plot for survival per gender
        sns.countplot(x='Sex', hue='Survived', data=df, palette=['coral', 'lightgreen'], ax=ax)

        # Add a title to the plot
        ax.set_title('Survival Chances per Gender')

        # Display the plot using Streamlit
        st.subheader('Overlevingskans per geslacht')
        st.pyplot(fig)
