def col1_age(st):
    st.subheader('Age Column')
    code = """
# er zijn nog steeds missende waarden in de Age kolom
# we kunnen deze vervangen door de gemiddelde leeftijd van de passagiers te nemen    
mean_age = df['Age'].mean()
df['Age'] = df['Age'].fillna(mean_age)

mean_age = df_test['Age'].mean()
df_test['Age'] = df_test['Age'].fillna(mean_age)
"""
    st.code(code, language='python')
    code = """
# groepeer de leeftijden in bins van 10 jaar    
bins = [0, 18, 30, 40, 50, 60, 70, 80]
df['Age_group'] = pd.cut(df['Age'], bins)
df_test['Age_group'] = pd.cut(df_test['Age'], bins)
"""
    st.code(code, language='python')

def col1_fare(st):
    st.subheader('Fare Column')
    code = """
# Kolom 'Fare' heeft 1 onbekende waarde in de test data, dus we vervangen deze door de gemiddelde waarde
mean_fare = df_test['Fare'].mean()
df_test['Fare'] = df_test['Fare'].fillna(mean_fare)
"""
    st.code(code, language='python')
    code = """
# groepeer de ticket prijzen in bins. de max is 513
bins = [0, 5, 10, 15, 20, 25, 50, 100, 200, 300, 513]
df['Fare_group'] = pd.cut(df['Fare'], bins)
df_test['Fare_group'] = pd.cut(df_test['Fare'], bins)
"""
    st.code(code, language='python')
    code = """
# Blijkbaar zijn er nu nog steeds missende waardes in de Fare_group kolom, dus we vullen deze in met de meest voorkomende waarde
df['Fare_group'] = df['Fare_group'].fillna(df['Fare_group'].mode()[0])
df_test['Fare_group'] = df_test['Fare_group'].fillna(df_test['Fare_group'].mode()[0])

# print de value counts van de Fare_group kolom
print(df['Fare_group'].value_counts())
"""

def col1_cabin(st):
    st.subheader('Cabin Column')
    code = """
# verwijder de 'Cabin' kolom, want deze bevat te veel missende waardes
df = df.drop('Cabin', axis=1)
df_test = df_test.drop('Cabin', axis=1)
"""
    st.code(code, language='python')

def col1_embarked(st):
    st.subheader('Embarked Column')
    code = """
# er zijn twee missende waardes in de Test dataset. In 'Embarked'. Deze kolom is erg belangrijk voor de voorspelling.
# Omdat het er maar twee zijn voegen we de meest voorkomende waarde toe.
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
"""
    st.code(code, language='python')