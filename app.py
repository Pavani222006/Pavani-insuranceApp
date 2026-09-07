import streamlit as st
import numpy as np
import pickle

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Health Insurance Cost Prediction",
    page_icon="🏥",
    layout="centered"
)

# -------------------------------------------------
# Custom Dark UI
# -------------------------------------------------
st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0b0f14, #151b24);
        color: white;
    }

    /* Main content */
    .block-container {
        max-width: 850px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    /* Title */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 8px;
        color: #ffffff;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #b8c1cc;
        margin-bottom: 35px;
    }

    /* Section headings */
    .section-title {
        font-size: 22px;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 18px;
        color: #ffffff;
    }

    /* Input labels */
    label {
        color: #e6e6e6 !important;
        font-weight: 500 !important;
    }

    /* Input boxes */
    div[data-baseweb="input"] {
        background-color: #202731;
        border-radius: 8px;
    }

    div[data-baseweb="select"] > div {
        background-color: #202731;
        border-radius: 8px;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        height: 52px;
        border-radius: 10px;
        border: none;
        background: linear-gradient(90deg, #2563eb, #3b82f6);
        color: white;
        font-size: 18px;
        font-weight: 600;
        margin-top: 20px;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #1d4ed8, #2563eb);
        color: white;
    }

    /* Result box */
    .result-box {
        background: #17202b;
        border: 1px solid #3b82f6;
        border-radius: 12px;
        padding: 20px;
        margin-top: 25px;
        text-align: center;
    }

    .result-title {
        font-size: 16px;
        color: #aeb8c4;
    }

    .result-value {
        font-size: 30px;
        font-weight: 700;
        color: #60a5fa;
        margin-top: 5px;
    }

    /* Info box */
    .info-box {
        background: #151c25;
        border-radius: 10px;
        padding: 15px;
        margin-top: 25px;
        text-align: center;
        color: #aeb8c4;
        font-size: 14px;
    }

</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# Load Model
# -------------------------------------------------
with open("New_RFmodel.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler if used
try:
    with open("New_scalar.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    scaler = None


# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown(
    '<div class="main-title">🏥 Health Insurance Cost Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Enter customer details to estimate the insurance cost using a machine learning model.'
    '</div>',
    unsafe_allow_html=True
)


# -------------------------------------------------
# Customer Details
# -------------------------------------------------
st.markdown(
    '<div class="section-title">👤 Customer Details</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "🎂 Age",
        min_value=0,
        max_value=100,
        value=30
    )

    bmi = st.number_input(
        "⚖️ BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )

    children = st.number_input(
        "👶 Number of Children",
        min_value=0,
        max_value=5,
        value=0
    )


with col2:

    gender = st.selectbox(
        "⚥ Gender",
        ["Female", "Male"]
    )

    smoker = st.selectbox(
        "🚬 Smoker",
        ["No", "Yes"]
    )

    region = st.selectbox(
        "📍 Region",
        ["northeast", "northwest", "southeast", "southwest"]
    )


# -------------------------------------------------
# Manual Encoding
# -------------------------------------------------

gender_male = 1 if gender == "Male" else 0

smoker_yes = 1 if smoker == "Yes" else 0

region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0

# northeast → all zeros


# -------------------------------------------------
# Combine Inputs
# -------------------------------------------------

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


# -------------------------------------------------
# Apply Scaling
# -------------------------------------------------

if scaler is not None:
    input_data = scaler.transform(input_data)


# -------------------------------------------------
# Prediction
# -------------------------------------------------

st.markdown("---")

if st.button("🔮 Predict Insurance Cost"):

    prediction = model.predict(input_data)

    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-title">Estimated Insurance Cost</div>
            <div class="result-value">₹ {prediction[0]:,.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------------------------------------
# Footer
# -------------------------------------------------

st.markdown(
    """
    <div class="info-box">
        📊 This application uses a Machine Learning Regression Model
        to estimate health insurance costs based on customer details.
    </div>
    """,
    unsafe_allow_html=True
)
