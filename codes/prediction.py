def prediction(st):
    col1, col2 = st.columns([1, 1])
    with col1:
        code="""
# Definieer de target-variabele (y) en de features (X) uit de train-set
y = train_data["Survived"]

features = ['Sex', 'Fare_group', 'Age_group', 'Embarked', 'Pclass', 'SibSp', 'Parch']

# Gebruik get_dummies om categorische variabelen zoals 'Sex' om te zetten naar numeriek
X_train = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

# Bouw en train het model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X_train, y)

# Maak voorspellingen op basis van de test-set
predictions = model.predict(X_test)

# print de prediction scores
print(model.score(X_train, y))

# Maak een output DataFrame met PassengerId en de voorspelde Survived-waarden
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
"""
        st.code(code, language='python')
        st.write("0.8282828282828283")
    with col2:
        code="""
features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Deck','is_alone','Title']

X_train = df_train[features]
Y_train = df_train['Survived']
X_Test = df_test.drop(['PassengerId'], axis= 1).copy()
X_Test = X_Test[features]
"""
        st.code(code, language='python')

        code="""
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_Test)

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
"""
        st.code(code, language='python')

        code="""
results = pd.DataFrame({'Model': ['Random Forest'],\
                       'Score': [acc_random_forest]})
"""
        st.code(code, language='python')
        st.write("""
                Model
Score	
92.7	Random Forest
""")
