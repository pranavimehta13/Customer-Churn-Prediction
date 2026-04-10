import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the enocoders and the scaler
with open('label_encoder_gender.pkl', 'rb') as f:
    le_gender = pickle.load(f)
    print(le_gender.classes_)
with open('onehot_encoder_geo.pkl', 'rb') as f:
    ohe_geo = pickle.load(f)    
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Streanlit app
st.title("Customer Churn Prediction")

# User Input
geography = st.selectbox('Geography', ohe_geo.categories_[0])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18, 100, 30)
balance = st.number_input('Balance', min_value=0.0, step=0.01)
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10, 3)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

le_gender.fit(['Male', 'Female'])

if st.button('Predict Churn'):
    # prepare the data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [le_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode the geography
    geo_encoded = ohe_geo.transform([[geography]])
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # scale only numeric columns
    numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    scaled_numeric = scaler.transform(input_data[numerical_features])

    # keep the other already-encoded columns
    other_features = input_data.drop(columns=numerical_features).to_numpy()

    # combine to the full input vector in the same order that the model expects
    full_input = np.concatenate([scaled_numeric, other_features], axis=1)

    # Predict churn probability
    prediction = model.predict(full_input)
    churn_prob = prediction[0][0]

    if churn_prob > 0.5:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is unlikely to churn.")