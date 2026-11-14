import os
import joblib
import pandas as pd
import streamlit as st

# The model file is expected in the same folder as this script (Downloads)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agentic ai.pkl")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("Sales Predictor")
st.write("Enter your advertising spend to predict sales.")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

youtube = st.number_input("YouTube spend", min_value=0.0, value=100.0, step=10.0)
facebook = st.number_input("Facebook spend", min_value=0.0, value=50.0, step=10.0)
newspaper = st.number_input("Newspaper spend", min_value=0.0, value=30.0, step=10.0)

if st.button("Predict sales"):
    new_data = pd.DataFrame(
        [[youtube, facebook, newspaper]],
        columns=["youtube", "facebook", "newspaper"],
    )
    prediction = model.predict(new_data)[0]
    st.success(f"Predicted sales: {prediction:.2f}")
