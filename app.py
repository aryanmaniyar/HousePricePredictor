# app.py

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Model load and training
housing = fetch_california_housing(as_frame=True)
df = housing.frame
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("🏠 California House Price Predictor")

st.write("Enter the details to estimate the house price:")

MedInc = st.slider('Median Income (10k USD)', 0.0, 15.0, 3.0)
HouseAge = st.slider('House Age', 1, 100, 30)
AveRooms = st.slider('Average Rooms', 1.0, 10.0, 5.0)
AveBedrms = st.slider('Average Bedrooms', 0.5, 5.0, 1.0)
Population = st.slider('Population', 100, 5000, 1500)
AveOccup = st.slider('Average Occupants per Household', 1.0, 5.0, 3.0)
Latitude = st.slider('Latitude', 32.0, 42.0, 36.0)
Longitude = st.slider('Longitude', -124.0, -114.0, -120.0)

input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])

if st.button("Predict Price"):
    prediction = model.predict(input_data)
    st.success(f"Estimated House Price: ${round(prediction[0]*100000, 2)}")
