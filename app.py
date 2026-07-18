# Q1: Import libraries
import streamlit as st
import pandas as pd
import joblib

# Q2: Load saved model and preprocessing objects
model = joblib.load("LR_model.pkl")        # trained Linear Regression model
scaler = joblib.load("scaler.pkl")         # StandardScaler
encoded_columns = joblib.load("columns.pkl")  # training columns after encoding

# Q3: Page configuration
st.set_page_config(page_title="Ford Car Price Predictor", layout="centered")

# Q4: Title and description
st.title("Ford Car Price Predictor")
st.write("Enter the car details below to predict its selling price.")

# Q5: Numerical inputs
year = st.number_input("Manufacturing Year", min_value=1990, max_value=2030, value=2019)
mileage = st.number_input("Mileage", min_value=0, max_value=200000, value=10000)
tax = st.number_input("Road Tax (£)", min_value=0, max_value=600, value=150)
mpg = st.number_input("Miles per Gallon (MPG)", min_value=0.0, max_value=250.0, value=55.0)
engineSize = st.number_input("Engine Size (L)", min_value=0.0, max_value=6.0, value=1.2)

# Q6: Dropdowns for categorical inputs
transmission = st.selectbox("Transmission", ["Manual", "Automatic", "Semi-Auto"])
fuelType = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Other"])

# Q7: Text input and Predict button
model_name = st.text_input("Car Model Name")
predict_btn = st.button("Predict Price")

# Q8–Q9: Prediction logic
if predict_btn:
    # Create DataFrame from user inputs
    input_data = pd.DataFrame({
        "model": [model_name],
        "year": [year],
        "price": [0],  # placeholder
        "transmission": [transmission],
        "mileage": [mileage],
        "fuelType": [fuelType],
        "tax": [tax],
        "mpg": [mpg],
        "engineSize": [engineSize]
    })

    # One-hot encoding
    input_encoded = pd.get_dummies(input_data, columns=["model", "transmission", "fuelType"])
    input_encoded = input_encoded.reindex(columns=encoded_columns, fill_value=0)

    # Scaling numerical columns
    num_cols = ["year", "mileage", "tax", "mpg", "engineSize"]
    input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])

    # Prediction
    prediction = model.predict(input_encoded)[0]

    # Output
    st.success(f"Predicted Price: £{prediction:.2f}")
