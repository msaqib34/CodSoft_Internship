import streamlit as st
import pickle
import numpy as np

# Load the trained model
app= open('model.pkl', 'rb')
model = pickle.load(app)

# Define the UI
st.title('Titanic Survival Prediction')

# Input widgets for user inputs
pclass = st.selectbox('Pclass', [1, 2, 3])
sex = st.radio('Sex', ['male', 'female'])
age = st.slider('Age', min_value=0, max_value=100, step=1)
fare = st.number_input('Fare', min_value=0.0)
embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])
family_size = st.slider('Family Size', min_value=0, max_value=10, step=1)
title = st.selectbox('Title', ['Mr', 'Mrs', 'Miss', 'Master', 'Don', 'Rev', 'Dr', 'Mme', 'Ms', 'Major', 'Lady', 'Sir', 'Mlle', 'Col', 'Capt', 'Countess', 'Jonkheer'])


# Convert user inputs to numerical values
sex_num = sex
embarked_num = embarked
title_num = title

# Convert user input data to NumPy array
user_data = np.array([[pclass, sex_num, age, fare, embarked_num, family_size, title_num]])

# Prediction button
if st.button('Predict'):
    # Make prediction
    prediction = model.predict(user_data)[0]

    # Display prediction
    st.write('Predicted Survival:', 'Survived' if prediction == 1 else 'Not Survived')
