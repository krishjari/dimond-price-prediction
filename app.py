
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------
# Page configuration
# --------------------------------
st.set_page_config(
    page_title="Diamond Price Predictor",
    page_icon="💎",
    layout="wide"
)

# --------------------------------
# Load saved model files
# --------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "diamond_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "diamond_scaler.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "diamond_columns.pkl")

@st.cache_resource
def load_model_files():
    for path in [MODEL_PATH, SCALER_PATH, COLUMNS_PATH]:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Required file not found: {os.path.basename(path)}"
            )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    saved_columns = joblib.load(COLUMNS_PATH)

    return model, scaler, list(saved_columns)


try:
    model, scaler, all_columns = load_model_files()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.info(
        "Check that all three .pkl files are present in the "
        "GitHub repository and available to Streamlit Cloud."
    )
    st.stop()

# --------------------------------
# Prepare training columns
# --------------------------------
TARGET = "price"

# The columns file may contain price or only input features.
if TARGET in all_columns:
    feature_columns = [
        col for col in all_columns if col != TARGET
    ]
else:
    feature_columns = all_columns

# True only if your training target (price) was scaled.
TARGET_WAS_SCALED = True

# --------------------------------
# Styling
# --------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1220, #17243a);
    color: #f5f7fb;
}
h1, h2, h3, p, label {
    color: #f5f7fb;
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
</style>
""", unsafe_allow_html=True)

# --------------------------------
# App heading
# --------------------------------
st.title("💎 Diamond Price Predictor")
st.write(
    "Enter the diamond details below to estimate its price "
    "using a trained machine-learning model."
)
st.divider()

# --------------------------------
# Input form
# --------------------------------
with st.form("diamond_form"):
    st.subheader("Diamond Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        carat = st.number_input(
            "Carat",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.01
        )
        cut = st.selectbox(
            "Cut",
            ["Fair", "Good", "Very Good", "Premium", "Ideal"]
        )

    with col2:
        depth = st.number_input(
            "Depth (%)",
            min_value=30.0,
            max_value=80.0,
            value=61.5,
            step=0.1
        )
        color = st.selectbox(
            "Color",
            ["D", "E", "F", "G", "H", "I", "J"]
        )

    with col3:
        table = st.number_input(
            "Table (%)",
            min_value=40.0,
            max_value=80.0,
            value=57.0,
            step=0.1
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
            "Length (x)",
            min_value=0.0,
            value=6.0,
            step=0.1
        )

    with c2:
        y = st.number_input(
            "Width (y)",
            min_value=0.0,
            value=6.0,
            step=0.1
        )

    with c3:
        z = st.number_input(
            "Height (z)",
            min_value=0.0,
            value=3.7,
            step=0.1
        )

    submitted = st.form_submit_button("💎 Predict Price")

# --------------------------------
# Prediction
# --------------------------------
if submitted:
    if x <= 0 or y <= 0 or z <= 0:
        st.error("Please enter valid diamond dimensions.")
        st.stop()

    numeric_data = pd.DataFrame([{
        "carat": carat,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z
    }])

    categorical_data = pd.DataFrame([{
        "cut": cut,
        "color": color,
        "clarity": clarity
    }])

    encoded_data = pd.get_dummies(categorical_data)

    input_data = pd.concat(
        [numeric_data, encoded_data],
        axis=1
    )

    # Match the exact columns and order used during training.
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    try:
        scaler_count = getattr(
            scaler, "n_features_in_", None
        )

        # Case 1: scaler was fitted on the full DataFrame,
        # including the price column.
        if (
            TARGET in all_columns
            and scaler_count == len(all_columns)
        ):
            feature_indices = [
                all_columns.index(col)
                for col in feature_columns
            ]

            input_scaled = (
                input_data.to_numpy(dtype=float)
                - scaler.mean_[feature_indices]
            ) / scaler.scale_[feature_indices]

        # Case 2: scaler was fitted only on input features.
        elif scaler_count == len(feature_columns):
            input_scaled = scaler.transform(input_data)

        else:
            st.error(
                "The saved scaler and columns do not match. "
                "Please check how the scaler and columns were "
                "saved during model training."
            )
            st.stop()

        prediction = float(
            np.asarray(
                model.predict(input_scaled)
            ).reshape(-1)[0]
        )

        # Reverse target scaling only when the saved scaler
        # includes the price column.
        if TARGET_WAS_SCALED:
            if (
                TARGET in all_columns
                and scaler_count == len(all_columns)
            ):
                price_index = all_columns.index(TARGET)

                prediction = (
                    prediction * scaler.scale_[price_index]
                    + scaler.mean_[price_index]
                )
            else:
                st.error(
                    "Your code says price was scaled, but the "
                    "saved scaler does not contain the price "
                    "column. Update the target scaling setup "
                    "to match your training notebook."
                )
                st.stop()

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

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Please check that your saved model, scaler, "
            "and training columns match."
        )
