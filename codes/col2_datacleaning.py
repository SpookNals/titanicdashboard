def col2_age(st):
    st.subheader('Leeftijd Kolom')
    code = """
# features voor het voorspellen van de leeftijd
features = ['Pclass','SibSp','Parch' ]

data = [df_train, df_test]
for dataset in data:

    # split de dataset in 2: een met leeftijden en een zonder
    age_train = dataset[dataset['Age'].notnull()]
    age_test = dataset[dataset['Age'].isnull()]

    # train het model op de rijen waar de leeftijd bekend is
    X_train = age_train[features]
    y_train = age_train['Age']

    # defineer en fit het model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # voorspel de missende leeftijden
    X_test = age_test[features]
    age_test['Age'] = rf.predict(X_test)

    # combineer de data
    dataset.loc[dataset['Age'].isnull(), 'Age'] = age_test['Age']

df_train = data[0]
df_test = data[1]
"""
    st.code(code, language='python')
    code = """
bins = [0, 11, 18, 22, 27, 33, 40, 66, float('inf')]
labels = [0, 1, 2, 3, 4, 5, 6, 7]
          
data = [df_train, df_test]
for dataset in data:
    dataset['Age'] = pd.cut(dataset['Age'], bins=bins, labels=labels, right=False)
    dataset['Age'] = dataset['Age'].astype(int)

df_train = data[0]
df_test = data[1]
"""
    st.code(code, language='python')

def col2_fare(st):
    st.subheader('Ticket Prijs Kolom')
    code = """
data = [df_train, df_test]
for dataset in data:


    median_fare_by_pclass = dataset.groupby('Pclass')['Fare'].median()

    def fill_fare(row):
        if pd.isnull(row['Fare']):
            return median_fare_by_pclass[row['Pclass']]
        else:
            return row['Fare']

    # Apply de functie om de missende waarden in te vullen
    dataset['Fare'] = dataset.apply(fill_fare, axis=1)

df_train = data[0]
df_test = data[1]
"""
    st.code(code, language='python')
    code = """
data = [df_train, df_test]
for dataset in data:
    dataset['Fare'] = dataset['Fare'].astype(int)

df_train = data[0]
df_test = data[1]   
"""
    st.code(code, language='python')
    code = """
fare_bins = [0, 7.91, 14.454, 31, 99, 250, float('inf')]
fare_labels = [0, 1, 2, 3, 4, 5]  # Corresponding labels for the bins

data = [df_train, df_test]
for dataset in data:
    dataset['Fare'] = pd.cut(dataset['Fare'], bins=fare_bins, labels=fare_labels, right=False)
    dataset['Fare'] = dataset['Fare'].astype(int)

df_train = data[0]
df_test = data[1]
"""
    st.code(code, language='python')

def col2_cabin(st):
    st.subheader('Kajuit Kolom')

    code = """
data = [df_train, df_test]

for dataset in data:
    # pak de eerste letter van de cabin
    dataset['Deck'] = dataset['Cabin'].str[0]

    # vul de missende waarden in met 'Unknown'
    dataset['Deck'] = dataset['Deck'].fillna('Unknown')

    # maak een nieuwe kolom Fare_bin aan
    dataset['Fare_bin'] = pd.qcut(dataset['Fare'], 8, labels=False, duplicates='drop')

    # Create a dictionary that holds the mode of Deck by Pclass and Fare_bin
    deck_medians = dataset[dataset['Deck'] != 'Unknown'].groupby(['Pclass', 'Fare_bin'])['Deck'].agg(lambda x: x.mode()[0]).to_dict()

    # Function to impute Deck based on Pclass and Fare_bin
    def fill_missing_deck(row, deck_medians):
        if row['Deck'] == 'Unknown':
            # Use tuple of (Pclass, Fare_bin) as the key to access the dictionary
            return deck_medians.get((row['Pclass'], row['Fare_bin']), 'U')  # Default to 'U' if no match
        else:
            return row['Deck']

    # Apply the function to fill missing Deck values
    dataset['Deck'] = dataset.apply(lambda row: fill_missing_deck(row, deck_medians), axis=1)

    # Drop the temporary 'Fare_bin' column after imputation
    dataset.drop(columns=['Fare_bin'], inplace=True)

    # Prepare data for model-based imputation of Deck
    # Use the rows where Deck is known for training the model
    train_data = dataset[dataset['Deck'] != 'U']
    test_data_unknown = dataset[dataset['Deck'] == 'U']

    # Define features and target
    features = ['Fare', 'Age']

    # Train a Random Forest Classifier to predict Deck
    X_train = train_data[features]
    y_train = train_data['Deck']

    # Initialize and fit the model
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Predict the missing Deck values in the test set
    X_test = test_data_unknown[features]
    test_data_unknown.loc[:, 'Deck'] = rf_classifier.predict(X_test)

    # Combine back the imputed values into the original dataset
    dataset.loc[dataset['Deck'] == 'U', 'Deck'] = test_data_unknown['Deck']


df_train = data[0]
df_test = data[1]
"""
    st.code(code, language='python')

    code = """
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}

data = [df_train, df_test]
for dataset in data:

    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].astype(int)
    # drop cabin
    dataset.drop(['Cabin'], axis=1, inplace=True)

df_train = data[0]
df_test = data[1]
"""
    st.code(code, language='python')

def col2_embarked(st):
    st.subheader('Opstapplaats Kolom')
    code = """
data = [df_train, df_test]


for dataset in data:

    embarked_mode = dataset.groupby('Pclass')['Embarked'].agg(lambda x: x.mode()[0])


    def fill_embarked(row):
        if pd.isnull(row['Embarked']):
            return embarked_mode[row['Pclass']]
        return row['Embarked']
    
    dataset['Embarked'] = dataset.apply(fill_embarked, axis=1)

df_train = data[0]
df_test = data[1]
"""
    st.code(code, language='python')

def col2_relatives(st):
    st.subheader('Familie Kolommen')
    code = """
# sibssp and parch
data = [df_train, df_test]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'is_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'is_alone'] = 1
    dataset['is_alone'] = dataset['is_alone'].astype(int)

df_train = data[0]
df_test = data[1]
"""
    st.code(code, language='python')

def col2_title(st):
    st.subheader('Aanspreek Titel Kolom')
    code = """
data = [df_train, df_test]
for dataset in data:
    dataset['Title'] = dataset['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)


# Define a mapping dictionary for titles
title_mapping = {
    "Mr": 1,
    "Miss": 2,
    "Mrs": 3,
    "Master": 4,
    "Vip": 5
}

# Define a function to map titles
def map_titles(title):
    if title in title_mapping:
        return title_mapping[title]
    elif title in ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']:
        return title_mapping["Vip"]
    elif title == 'Mlle' or title == 'Ms':
        return title_mapping["Miss"]
    elif title == 'Mme':
        return title_mapping["Mrs"]
    else:
        return 0  # Default for any unhandled titles

# Apply the mapping function to the Title column
for dataset in data:
    dataset['Title'] = dataset['Title'].apply(map_titles)


df_train = data[0]
df_test = data[1]
"""
    st.code(code, language='python')

def col2_numeric_features(st):
    st.subheader('Numerieke Features')
    code = """
genders = {'male': 0, 'female': 1}
ports = {"S": 0, "C": 1, "Q": 2}

data = [df_train, df_test]

for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(genders)
    dataset['Embarked'] = dataset['Embarked'].map(ports)


df_train = data[0]
df_test = data[1]
"""
    st.code(code, language='python')	
