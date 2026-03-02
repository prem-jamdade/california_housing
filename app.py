import streamlit as st
import pickle
import numpy as np

# Load model
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("California House Price Predictor")

st.write("Enter house details below:")

MedInc = st.number_input("Median Income")
HouseAge = st.number_input("House Age")
AveRooms = st.number_input("Average Rooms")
AveBedrms = st.number_input("Average Bedrooms")
Population = st.number_input("Population")
AveOccup = st.number_input("Average Occupancy")
Latitude = st.number_input("Latitude")
Longitude = st.number_input("Longitude")

if st.button("Predict Price"):
    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                          Population, AveOccup, Latitude, Longitude]])

    prediction = model.predict(features) * 100000

    st.success(f"Predicted House Price: {prediction[0]:.2f} USD")
