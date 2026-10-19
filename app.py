
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------
# Load saved files
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "diamond_model.pkl")
scaler_path = os.path.join(BASE_DIR, "diamond_scaler.pkl")
columns_path = os.path.join(BASE_DIR, "diamond_columns.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
feature_columns = joblib.load(columns_path)

TARGET = "price"

# Change this to False if y_train was NOT scaled
TARGET_WAS_SCALED = True

feature_columns = [
    col for col in all_columns if col != TARGET
]

numeric_columns = [
    "carat", "depth", "table", "x", "y", "z"
]

categorical_columns = ["cut", "color", "clarity"]

# -------------------------------
# Page setup
# -------------------------------
st.set_page_config(
    page_title="Diamond Price Predictor",
    page_icon="💎",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1220, #17243a);
    color: #f5f7fb;
}
h1, h2, h3, p, label {
    color: #f5f7fb !important;
}
div.stButton > button {
    background-color: #39d5c5;
    color: #081321;
    border-radius: 10px;
    font-weight: bold;
    width: 100%;
    height: 48px;
    border: none;
}
div.stButton > button:hover {
    background-color: #7ceadd;
    color: #081321;
}
</style>
""", unsafe_allow_html=True)

st.title("💎 Diamond Price Predictor")
st.write(
    "Enter the diamond details below to estimate its price "
    "using a trained Random Forest model."
)

st.divider()

# -------------------------------
# Input form
# -------------------------------
with st.form("diamond_form"):

    st.subheader("Diamond Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        carat = st.number_input(
            "Carat", min_value=0.01,
            max_value=10.0, value=1.0, step=0.01
        )

        cut = st.selectbox(
            "Cut",
            ["Fair", "Good", "Very Good", "Premium", "Ideal"]
        )

    with col2:
        depth = st.number_input(
            "Depth (%)", min_value=30.0,
            max_value=80.0, value=61.5, step=0.1
        )

        color = st.selectbox(
            "Color",
            ["D", "E", "F", "G", "H", "I", "J"]
        )

    with col3:
        table = st.number_input(
            "Table (%)", min_value=40.0,
            max_value=80.0, value=57.0, step=0.1
        )

        clarity = st.selectbox(
            "Clarity",
            ["I1", "IF", "SI1", "SI2",
             "VS1", "VS2", "VVS1", "VVS2"]
        )

    st.subheader("Diamond Dimensions (mm)")

    c1, c2, c3 = st.columns(3)

    with c1:
        x = st.number_input(
            "Length (x)", min_value=0.0,
            value=6.0, step=0.1
        )

    with c2:
        y = st.number_input(
            "Width (y)", min_value=0.0,
            value=6.0, step=0.1
        )

    with c3:
        z = st.number_input(
            "Height (z)", min_value=0.0,
            value=3.7, step=0.1
        )

    submitted = st.form_submit_button("💎 Predict Price")

# -------------------------------
# Prediction
# -------------------------------
if submitted:

    if x <= 0 or y <= 0 or z <= 0:
        st.error("Please enter valid diamond dimensions.")
        st.stop()

    # Numeric inputs
    input_data = pd.DataFrame([{
        "carat": carat,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z
    }])

    # One-hot encode categorical inputs
    categorical_data = pd.DataFrame([{
        "cut": cut,
        "color": color,
        "clarity": clarity
    }])

    encoded_data = pd.get_dummies(categorical_data)

    # Combine numeric and encoded columns
    input_data = pd.concat(
        [input_data, encoded_data], axis=1
    )

    # Match exact training columns and order
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Apply the SAME scaler statistics used in training.
    # This assumes scaler was fitted on the full encoded
    # DataFrame, including the price target.
    feature_indices = [
        all_columns.index(col) for col in feature_columns
    ]

    input_scaled = (
        input_data.to_numpy(dtype=float)
        - scaler.mean_[feature_indices]
    ) / scaler.scale_[feature_indices]

    # Predict
    prediction = float(
        np.asarray(model.predict(input_scaled)).reshape(-1)[0]
    )

    # Convert scaled target back to original price
    if TARGET_WAS_SCALED:
        price_index = all_columns.index(TARGET)

        prediction = (
            prediction * scaler.scale_[price_index]
            + scaler.mean_[price_index]
        )

    st.divider()
    st.subheader("Prediction Result")

    st.success("Price prediction completed!")

    st.metric(
        label="Estimated Diamond Price",
        value=f"{prediction:,.2f}"
    )

    st.caption(
        "This is a machine-learning estimate, "
        "not a guaranteed market price."
    )

    with st.expander("View entered diamond details"):
        st.dataframe(
            pd.DataFrame([{
                "Carat": carat,
                "Cut": cut,
                "Color": color,
                "Clarity": clarity,
                "Depth": depth,
                "Table": table,
                "X": x,
                "Y": y,
                "Z": z
            }]),
            use_container_width=True,
            hide_index=True
        )
