import joblib
import numpy as np
import pandas as pd
import streamlit as st

# set page title and layout
st.set_page_config(
    page_title = "Student Marks Predictor", 
    page_icon="🎓",
    layout = "centered"
)

st.title("🎓 Student Marks Predictor")
st.write("Enter study hours to predict the student's expected marks.")

# load saved ML Model
def load_model():
    return joblib.load("student_marks_model.pkl")

model = load_model()

hours = st.number_input(
    label = 'Study hours (per day)',
    min_value= 0.0,
    max_value=24.0,
    value=6.0,
    step=0.25,

)

if st.button("Predict Marks", type="primary"):
    if hours > 18:
        st.error("Cannot study more than 18 hours")
    elif hours <= 0:
        st.error("Please enter valid study hour")
    else:
        input_marks = pd.DataFrame({"study_hours" : [hours]})
        prediction = model.predict(input_marks)[0]
        #st.write(input_marks)
        #st.dataframe(input_marks)
        final_score = np.clip(prediction, 0.0, 100.0)
        st.success(f"### Predicted Marks: **{final_score.round(2)}**")