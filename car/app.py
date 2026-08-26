import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from PIL import Image

st.set_page_config(layout="wide")

#Load model
with open('linear_model.pkl', 'rb') as f:
    lm2 = pickle.load(f)

# Load feature importance from an Excel file
def load_feature_importance(file_path):
    return pd.read_excel(file_path)

# load feature important dataframe
final_fi = load_feature_importance('feature_importance.xlsx')

# Sidebar setup
image_sidebar = Image.open('Pic 1.png')
st.sidebar.image(image_sidebar, use_container_width=True)
st.sidebar.header('Vehicle Features')

# sidebar feature selection
def get_user_input():
    horsepower = st.sidebar.number_input('Horsepower', min_value=0, max_value=1000, step=1, value=300)
    torque = st.sidebar.number_input('Torque', min_value=0, max_value=1500, step=1, value=400)
    make = st.sidebar.selectbox('Make',['Aston Martin', 'Audi', 'BMW', 'Bentley', 'Ford', 'Mercedes-Benz', 'Nissan'])
    body_size = st.sidebar.selectbox('Body Size',['Compact', 'Large', 'Midsize'])
    body_style = st.sidebar.selectbox('Body Style',
    [
        'Cargo Minivan', 'Cargo Van', 'Convertible', 'Convertible SUV', 'Coupe', 'Hatchback', 
        'Passenger Minivan', 'Passenger Van', 'Pickup Truck', 'SUV', 'Sedan', 'Wagon'
    ])
    engine_aspiration = st.sidebar.selectbox('Engine Aspiration', [
        'Electric Motor', 'Naturally Aspirated', 'Supercharged', 'Turbocharged', 'Twin-Turbo', 'Twincharged'
    ])
    drivetrain = st.sidebar.selectbox('Drivetrain', ['4WD', 'AWD', 'FWD', 'RWD'])
    transmission = st.sidebar.selectbox('Transmission', ['automatic', 'manual'])

    user_data = {
        'Horsepower': horsepower,
        'Torque':torque,
        f'Make_{make}': 1,
        f'Body Size_{body_size}': 1,
        f'Body Style_{body_style}': 1,
        f'Engine Aspiration_{engine_aspiration}': 1,
        f'Drivetrain_{drivetrain}': 1,
        f'Transmission_{transmission}': 1,

    }
    return user_data

# Top banner
image_banner = Image.open('pic2.jpg')  # Replace with your image file
resized_image = image_banner.resize((800, 200))
st.image(resized_image)

st.markdown("<h1 style='text-align: center;'>Vehicle Price Prediction App</h1>", unsafe_allow_html=True)

left_col, right_col = st.columns(2)

with left_col:
    st.header('Feature importance')
    final_fi_sorted = final_fi.sort_values(by='Feature Importance Score', ascending=True)

    # Create interactive bar chart with Plotly
    fig = px.bar(
        final_fi_sorted,
        x='Feature Importance Score',
        y='Variable',
        orientation='h',
        title="Feature Importance",
        labels={'Feature Importance Score': 'Importance', 'Variable': 'Feature'},
        text='Feature Importance Score',
        color_discrete_sequence=['#48a3b4']  # Custom bar color
    )
    fig.update_layout(
        xaxis_title="Feature Importance Score",
        yaxis_title="Variable",
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    with right_col:
        st.header("Predict Vehicle Price")

        # User inputs from sidebar
        user_data = get_user_input()

        # Transform the input into the required format
        def prepare_input(data, feature_list):
            input_data = {feature: data.get(feature, 0) for feature in feature_list}
            return np.array([list(input_data.values())])

        
        # Feature list (same order as used during model training)
        features = [
            'Horsepower', 'Torque', 'Make_Aston Martin', 'Make_Audi', 'Make_BMW', 'Make_Bentley',
            'Make_Ford', 'Make_Mercedes-Benz', 'Make_Nissan', 'Body Size_Compact', 'Body Size_Large',
            'Body Size_Midsize', 'Body Style_Cargo Minivan', 'Body Style_Cargo Van', 
            'Body Style_Convertible', 'Body Style_Convertible SUV', 'Body Style_Coupe', 
            'Body Style_Hatchback', 'Body Style_Passenger Minivan', 'Body Style_Passenger Van',
            'Body Style_Pickup Truck', 'Body Style_SUV', 'Body Style_Sedan', 'Body Style_Wagon',
            'Engine Aspiration_Electric Motor', 'Engine Aspiration_Naturally Aspirated',
            'Engine Aspiration_Supercharged', 'Engine Aspiration_Turbocharged',
            'Engine Aspiration_Twin-Turbo', 'Engine Aspiration_Twincharged', 
            'Drivetrain_4WD', 'Drivetrain_AWD', 'Drivetrain_FWD', 'Drivetrain_RWD', 
            'Transmission_automatic', 'Transmission_manual'
        ]

        if st.button("Predict"):
            input_array = prepare_input(user_data, features)
            prediction = lm2.predict(input_array)
            st.subheader("Predicted Price")
            st.write(f"${prediction[0]:,.2f}")