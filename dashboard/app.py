import streamlit as st
import joblib
import numpy as np

st.title("Credit Card Fraud Detection")

amount = st.number_input("Transaction Amount")
v_features = [st.slider(f"V{i}", -30.0, 30.0, 0.0) for i in range(1, 29)]

if st.button("Predict"):
    model = joblib.load("models/random_forest.joblib")
    features = np.array([[*v_features, amount]])
    prediction = model.predict(features)
    st.write("Fraudulent" if prediction[0] == 1 else "Legitimate")
