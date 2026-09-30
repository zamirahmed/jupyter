import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.set_page_config(page_title="Breast Cancer Predition", layout="wide")
st.title("Cancer Prediction APP")

@st.cache_resource
def load_artifact():
    model = joblib.load('breast_cancer_logistic_model.pkl')
    scaler = joblib.load('breast_cancer_scaler.pkl')
    return model, scaler


try:
    model, scaler = load_artifact()
except FileNotFoundError:
    st.error(
        'Could not find Model file'
    )
    st.stop()


FEATURE_NAMES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension"
]

DEFAULT_VALUES = [
    14.1, 19.3, 91.9, 654.9, 0.096, 0.104, 0.089, 0.048, 0.181, 0.063,
    0.405, 1.217, 2.866, 40.34, 0.007, 0.025, 0.032, 0.012, 0.021, 0.0038,
    16.3, 25.7, 107.3, 880.6, 0.132, 0.254, 0.272, 0.115, 0.290, 0.084
]

#------------------------------------
# Choose sidebar
#-----------------------------------
st.sidebar.header("Input Method")
input_method = st.sidebar.radio(
    "choose how to provide patient data",
)

if input_method == 'Manual Entry':
    st.subheader("Enter Feature values")

    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]


    user_input = []
    for i, (feature, default) in enumerate(zip(FEATURE_NAMES, DEFAULT_VALUES)):
        col = columns[i % 3]
        val = col.number_input(
            label=feature,
            value = float(default),
            format="%.4f",
            key= feature

        )
        user_input.append(val)

    if st.button("Predict", type="primary"):
        input_array = np.array(user_input).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 0:
            st.error(f"Prediction: ** Maligent ** ")
        else:
            st.error(f"Prediction: ** Benign **")

        
        pred_col1, pred_col2 = st.columns(2)
        pred_col1.metric("Probality of Maligent: ", f"{proba[0]*100:.2f}%")
        pred_col2.metric("Probality of Benign: ", f"{proba[1]*100:.2f}%")

        st.progress(float(proba[1]))

else:
    st.subheader("Upload CSV")
    uploaded_file = st.file_uploader("Choose a CSV File", type="csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Preview of data file")
            st.dataframe(data.head())

            if st.button("Run Batch Prediction", type="primary"):
                data_scaled = scaler.transform(data)
                predictions = model.Predict(data_scaled)
                probabilities = model.predict_proba(data_scaled)[:,1]


                results = data.copy()
                results["Prediction"] = np.where(
                    predictions == 0, "Malignant", "Benign"
                )
                results["Probability_Benign"] = probabilities

                st.subheader("Results")
                st.dataframe(results)


                csv_out = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Result as CSV",
                    data=csv_out,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error {e}")

st.divider()
st.caption("Model: Logistic Regression | Train on Breast Cancer Dataset")