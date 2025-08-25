import streamlit as st
import pickle
import numpy as np
# Load the trained model
with open('titanic_model.pkl', 'rb') as file:
    model = pickle.load(file)
st.title('Titanic Survival Prediction App')
st.write('Enter passenger details to predict survival:')
# User inputs
Pclass = st.selectbox('Passenger Class (Pclass)', [1, 2, 3])
Sex = st.radio('Sex', ['male', 'female'])
Age = st.slider('Age', 0, 100, 25)
SibSp = st.number_input('Number of Siblings/Spouses Aboard', min_value=0, max_value=10, value=0)
Parch = st.number_input('Number of Parents/Children Aboard', min_value=0, max_value=10, value=0)
Fare = st.number_input('Fare', min_value=0.0, value=32.0)
Embarked = st.selectbox('Embarked Port', ['C', 'Q', 'S'])
# Encode categorical variables
Sex_encoded = 1 if Sex == 'male' else 0
Embarked_encoded = {'C':0, 'Q':1, 'S':2}[Embarked]
# Prepare input
X = np.array([[Pclass, Sex_encoded, Age, SibSp, Parch, Fare, Embarked_encoded]])
# Predict
if st.button('Predict'):
    prediction = model.predict(X)
    if prediction[0]==1:
        st.success('Passenger is likely to SURVIVE.')
    else:
        st.error('Passenger is NOT likely to survive')
