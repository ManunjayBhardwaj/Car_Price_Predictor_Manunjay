import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load the trained model and scaler with error handling
try:
    model = pickle.load(open('LinearRegression1.pkl', 'rb'))
    scaler = pickle.load(open('scaler1.pkl', 'rb'))
except FileNotFoundError as e:
    st.error("Error: Model or scaler file not found.")
    st.stop()

# Streamlit App Layout
st.title("🚗 Car Price Prediction")

# Add a short description to the app
st.markdown("""
    ### Enter the car details below to predict its price!
    This model predicts the selling price of a car based on various attributes.
""")

# Input Fields for the car features
fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
year = st.number_input("Car Year", min_value=1990, max_value=2025, step=1)
kms_driven = st.number_input("KMs Driven", min_value=0, step=1)
owner = st.number_input("Number of Previous Owners", min_value=0, step=1)
present_price = st.number_input("Present Price (₹)", min_value=0.0, step=0.1)

# Encoding the input values
fuel_type_encoded = {"Petrol": 0, "Diesel": 1, "CNG": 2}[fuel_type]
transmission_encoded = {"Manual": 0, "Automatic": 1}[transmission]
seller_type_encoded = {"Dealer": 0, "Individual": 1}[seller_type]

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'Year': [year],
    'Present_Price': [present_price],
    'Kms_Driven': [kms_driven],
    'Fuel_Type': [fuel_type_encoded],
    'Seller_Type': [seller_type_encoded],
    'Transmission': [transmission_encoded],
    'Owner': [owner]
})

# Ensuring the columns are in the same order as during model training
expected_columns = ['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']
input_data = input_data[expected_columns]  # Ensure the correct order

# Feature scaling
try:
    input_data_scaled = scaler.transform(input_data)
except ValueError as e:
    st.error(f"Error in feature scaling: {str(e)}")
    st.stop()

# Predict button
if st.button("🔮 Predict"):
    try:
        prediction = model.predict(input_data_scaled)
        st.subheader(f"Predicted Selling Price:")
        st.write(f"### ₹{prediction[0]:,.2f}")
        st.success("Prediction successful! 🚗💨")
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# Add a footer with some custom text
st.markdown("""
    ---
    Made with ❤️ by Manunjay 
    
    Data sourced from various car features.
""")
