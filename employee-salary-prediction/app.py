import streamlit as st
import numpy as np
import joblib

# Load the trained model pipeline
model = joblib.load("bestfin_nn_model.pkl")

st.title("Employee Salary Prediction")

# Mapping dictionaries (as before)
workclass_map = {
    "Private": 0, "Local-gov": 1, "?": 2, "Self-emp-not-inc": 3,
    "Federal-gov": 4, "State-gov": 5, "Self-emp-inc": 6,
    "Without-pay": 7, "Never-worked": 8
}
marital_status_map = {
    "Never-married": 0, "Married-civ-spouse": 1, "Widowed": 2,
    "Divorced": 3, "Separated": 4, "Married-spouse-absent": 5,
    "Married-AF-spouse": 6
}
occupation_map = {
    "Machine-op-inspct": 0, "Farming-fishing": 1, "Protective-serv": 2,
    "?": 3, "Other-service": 4, "Prof-specialty": 5, "Craft-repair": 6,
    "Adm-clerical": 7, "Exec-managerial": 8, "Tech-support": 9,
    "Sales": 10, "Priv-house-serv": 11, "Transport-moving": 12,
    "Handlers-cleaners": 13, "Armed-Forces": 14
}
relationship_map = {
    "Own-child": 0, "Husband": 1, "Not-in-family": 2,
    "Unmarried": 3, "Wife": 4, "Other-relative": 5
}
race_map = {
    "Black": 0, "White": 1, "Asian-Pac-Islander": 2,
    "Other": 3, "Amer-Indian-Eskimo": 4
}
gender_map = {
    "Male": 0, "Female": 1
}
native_country_map = {
    'United-States': 0, '?': 1, 'Peru': 2, 'Guatemala': 3, 'Mexico': 4,
    'Dominican-Republic': 5, 'Ireland': 6, 'Germany': 7, 'Philippines': 8,
    'Thailand': 9, 'Haiti': 10, 'El-Salvador': 11, 'Puerto-Rico': 12,
    'Vietnam': 13, 'South': 14, 'Columbia': 15, 'Japan': 16, 'India': 17,
    'Cambodia': 18, 'Poland': 19, 'Laos': 20, 'England': 21, 'Cuba': 22,
    'Taiwan': 23, 'Italy': 24, 'Canada': 25, 'Portugal': 26, 'China': 27,
    'Nicaragua': 28, 'Honduras': 29, 'Iran': 30, 'Scotland': 31,
    'Jamaica': 32, 'Ecuador': 33, 'Yugoslavia': 34, 'Hungary': 35,
    'Hong': 36, 'Greece': 37, 'Trinadad&Tobago': 38,
    'Outlying-US(Guam-USVI-etc)': 39, 'France': 40, 'Holand-Netherlands': 41
}

# Education mapping
education_num_map = {
    "Preschool": 1,
    "1st-4th Grade": 2,
    "5th-6th Grade": 3,
    "7th-8th Grade": 4,
    "9th Grade": 5,
    "10th Grade": 6,
    "11th Grade": 7,
    "12th Grade": 8,
    "High School Grad (HS-grad)": 9,
    "Some College": 10,
    "Associate-vocational": 11,
    "Associate-academic": 12,
    "Bachelor's Degree": 13,
    "Master's Degree": 14,
    "Professional School (Prof-school)": 15,
    "Doctorate (PhD)": 16
}

# Input fields
age = st.number_input("Age", min_value=18, max_value=75, value=30)
workclass = st.selectbox("Workclass", list(workclass_map.keys()))
fnlwgt = st.number_input("fnlwgt", min_value=0, value=100000)
education_level = st.selectbox("Education Level", list(education_num_map.keys()))
marital_status = st.selectbox("Marital Status", list(marital_status_map.keys()))
occupation = st.selectbox("Occupation", list(occupation_map.keys()))
relationship = st.selectbox("Relationship", list(relationship_map.keys()))
race = st.selectbox("Race", list(race_map.keys()))
gender = st.selectbox("Gender", list(gender_map.keys()))
capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40)
native_country = st.selectbox("Native Country", list(native_country_map.keys()))

# Convert selections to encoded values
features = np.array([[
    age,
    workclass_map[workclass],
    fnlwgt,
    education_num_map[education_level],  # <-- Added education-num here
    marital_status_map[marital_status],
    occupation_map[occupation],
    relationship_map[relationship],
    race_map[race],
    gender_map[gender],
    capital_gain,
    capital_loss,
    hours_per_week,
    native_country_map[native_country]
]])

st.write("Model input features:", features)

# Prediction
if st.button("Predict"):
    prediction = model.predict(features)
    st.write("Prediction:", "More than 50K" if prediction[0] == 1 else "50K or less")
    