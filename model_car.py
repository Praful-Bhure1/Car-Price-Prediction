import pandas as pd 
import numpy as np
import pickle
from sklearn import *
import streamlit as st 
import warnings
warnings.filterwarnings('ignore')

# Load the data and model
df = pickle.load(open('car_df.pkl', 'rb'))
model = pickle.load(open('rf.pkl', 'rb'))

def predict_price(brand, year, kmdriven, fuel, seller_type, transmission, owner):
    # Create a new dataframe with the user inputs
    inputs = pd.DataFrame([[brand, year, kmdriven, fuel, seller_type, transmission, owner]],
                          columns=['brand name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner'])

    # Check that the input data has the expected number of features
    if inputs.shape[1] != 7:
        raise ValueError(f'Expected 7 features but got {inputs.shape[1]}')

    # Make the prediction
    price = model.predict(inputs)

    return price[0]


# Create the user interface
st.title('Second hand Car Price Prediction')
st.header('Fill in the details to predict the price of your car')
brand = st.selectbox('Brand', df['brand name'].unique())
year = st.selectbox('Year', df['year'].unique())
kmdriven = st.number_input('km_driven', min_value=0)
fuel = st.selectbox('Fuel', ['Diesel', 'Petrol', 'CNG', 'LPG', 'Electric'])
seller_type = st.selectbox('Seller Type', ['Individual', 'Dealer', 'Trustmark Dealer'])
transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
owner = st.selectbox('Owner', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'])

# Add a button to trigger the prediction
if st.button('Predict Price'):
    price = predict_price(brand, year, kmdriven, fuel, seller_type, transmission, owner)
    st.success(f'The predicted price of the car is {price:,.0f} Indian Rupees.')