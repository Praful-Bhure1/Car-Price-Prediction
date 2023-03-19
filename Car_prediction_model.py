import streamlit as st
from sklearn import *
import pandas as pd
import numpy as np
import pickle
model = pickle.load(open('RF_price_predicting_model.pkl','rb'))


def main():
    st.title("Selling Price Predictor 🚗")
    st.markdown("##### Are you planning to sell your car !?\n##### So let's try evaluating the price.. 🤖 ")

    # @st.cache(allow_output_mutation=True)
    # def get_model():
    #     model = pickle.load(open('RF_price_predicting_model.pkl','rb'))
    #     return model

    st.write('')
    st.write('')

    years = st.number_input('In which year car was purchased ?',1996, 2020, step=1, key ='year')
    Years_old = 2020-years

    Present_Price = st.number_input('What is the current ex-showroom price of the car ?  (In ₹lakhs)', 0.00, 50.00, step=0.5, key ='present_price') 

    kms_driven = st.number_input('What is distance completed by the car in Kilometers ?', 0.00, 500000.00, step=500.00, key ='drived')

    owner = st.radio("The number of owners the car had previously ?", (0, 1, 3), key='owner')

    fuel_Petrol = st.selectbox('What is the fuel type of the car ?',('Petrol','Diesel', 'CNG'), key='fuel')
    if(fuel_Petrol=='Petrol'):
        fuel_Petrol=1
        fuel_Diesel=0
    elif(fuel_Petrol=='Diesel'):
        fuel_Petrol=0
        fuel_Diesel=1
    else:
        fuel_Petrol=0
        fuel_Diesel=0

    seller_type_Individual = st.selectbox('Are you a dealer or an individual ?', ('Dealer','Individual'), key='dealer')
    if(seller_type_Individual=='Individual'):
        seller_type_Individual=1
    else:
        seller_type_Individual=0	

    transmission_Mannual = st.selectbox('What is the transmission Type ?', ('Manual','Automatic'), key='manual')
    if(transmission_Mannual=='Mannual'):
        transmission_Mannual=1
    else:
        transmission_Mannual=0


    if st.button("Estimate Price", key='predict'):
        try:
            Model = model  #get_model()
            prediction = Model.predict([['selling_price', 'km_driven', 'Years_old', 'fuel_Diesel',
                                        'fuel_Electric', 'fuel_Petrol', 'seller_type_Individual',
                                        'seller_type_Trustmark Dealer', 'transmission_Manual',
                                        'owner_Fourth & Above Owner', 'owner_Second Owner',
                                        'owner_Test Drive Car', 'owner_Third Owner']])
            output = round(prediction[0],2)
            if output<0:
                st.warning("You will be not able to sell this car !!")
            else:
                st.success("You can sell the car for {} lakhs 🙌".format(output))
        except:
            st.warning("Opps!! Something went wrong\nTry again")
            
            