import streamlit as st
import numpy as np
import pickle

# Page configuration
st.set_page_config(
    page_title="Health Insurance Cost Prediction",
    page_icon="🏥",
    layout="centered"
)

# Dark UI styling
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }

    h1 {
        text-align: center;
        color: white;
        font-size: 42px;
    }

    .stMarkdown, label {
        color: white !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #262730 !important;
        color: white !important;
    }

    div[data-baseweb="select"] span {
        color: white !important;
    }

    input {
        background-color: #262730 !important;
        color: white !important;
    }

    button {
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# Load model
with open("New_RFmodel.pkl", "rb") as f:
    model = pickle.load(f)


# Load scaler if available
try:
    with open("New_scalar.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    scaler = None


# Title
st.title("Health Insurance Cost Prediction")

st.write("Enter the customer details to predict insurance cost")


# User inputs - one by one
age = st.number_input(
    "Age",
    min_value=0,
    max_value=100,
    value=30
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

children = st.number_input(
    "Number of Children",
    min_value=0,
    max_value=5,
    value=0
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

smoker = st.selectbox(
    "Smoker",
    ["No", "Yes"]
)

region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)


# Manual encoding
gender_male = 1 if gender == "Male" else 0

smoker_yes = 1 if smoker == "Yes" else 0

region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0


# Combine inputs
input_data = np.array([[
    age,
    bmi,
    children,
    gender_male,
    smoker_yes,
    region_northwest,
    region_southeast,
    region_southwest
]])


# Apply scaling if used
if scaler is not None:
    input_data = scaler.transform(input_data)


# Prediction button
if st.button("Predict Insurance Cost", use_container_width=True):

    prediction = model.predict(input_data)

    st.success(
        f"Estimated Insurance Cost: ₹ {prediction[0]:,.2f}"
    )
