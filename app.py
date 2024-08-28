import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import streamlit as st

# Load pre-trained model and encoders
model = load_model('model.h5')

with open('labelencoder.pkl', 'rb') as file:
    le = pickle.load(file)

with open('standardscaler.pkl', 'rb') as file:
    ss = pickle.load(file)

with open('onehotencoder.pkl', 'rb') as file:
    ohe = pickle.load(file)

# Streamlit app title
st.title('Customer Churn Prediction')

# User input fields
CreditScore = st.number_input('Enter CreditScore', min_value=0, step=100, value=0, placeholder='Credit Score')
Geography = st.selectbox('Select Geography', ohe.categories_[0])
Gender = st.selectbox('Select Gender', le.classes_)
Age = st.slider('Select Age', min_value=18, max_value=100, value=18)
Tenure = st.slider('Select Tenure', min_value=0, max_value=10, value=0)
Balance = st.number_input('Enter Balance', value=0.0, placeholder='Balance')
NumOfProducts = st.slider('Select Number of Products', min_value=0, max_value=4, value=0)
HasCrCard = st.selectbox('Has Credit Card?', [0, 1])
IsActiveMember = st.selectbox('Is Active Member?', [0, 1])
EstimatedSalary = st.number_input('Enter Estimated Salary', min_value=0, step=100, value=0, placeholder='Estimated Salary')

# Creating the input dictionary
input_dict = {
    'CreditScore': [CreditScore],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary]
}

# Encoding categorical features
encoded_gender = le.transform([Gender])
input_dict['Gender'] = encoded_gender

encoded_geo = ohe.transform([[Geography]]).toarray()
encoded_geo_df = pd.DataFrame(encoded_geo, columns=ohe.get_feature_names_out(['Geography']))

# Converting the input dictionary into a DataFrame
input_df = pd.DataFrame(input_dict)

# Concatenating the encoded geography DataFrame with the input DataFrame
final_df = pd.concat([input_df, encoded_geo_df], axis=1)

# Ensure the columns are in the correct order
expected_order = ss.feature_names_in_
final_df = final_df[expected_order]

# Scaling the final DataFrame
scaled_df = ss.transform(final_df)

# Making predictions
y_pred = model.predict(scaled_df)
y_pred_prob = y_pred[0][0]

# Displaying the result
if y_pred_prob > 0.5:
    st.write('The customer is likely to churn')
else:
    st.write('The customer is not likely to churn')
