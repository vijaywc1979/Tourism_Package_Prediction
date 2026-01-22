import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# -------- CONFIG --------
MODEL_REPO = "vijaywc1979/Tourism-Package-DecisionTree"
MODEL_FILE = "decision_tree_model.pkl"

# -------- LOAD MODEL FROM HF --------
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
model = joblib.load(model_path)

st.title("Wellness Tourism Package Predictor")
st.write("Predict whether a customer will purchase the Wellness Tourism Package.")

# -------- USER INPUT --------
age = st.number_input("Age", min_value=18, max_value=100, value=30)
typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
citytier = st.selectbox("City Tier", ["1", "2", "3"])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Others"])
gender = st.selectbox("Gender", ["Male", "Female"])
num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=1)
preferred_star = st.number_input("Preferred Hotel Star", min_value=1, max_value=7, value=3)
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
num_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=20, value=1)
passport = st.selectbox("Passport (Yes=1, No=0)", [0,1])
own_car = st.selectbox("Own Car (Yes=1, No=0)", [0,1])
num_children = st.number_input("Number of Children (<5)", min_value=0, max_value=5, value=0)
designation = st.text_input("Designation", "Employee")
monthly_income = st.number_input("Monthly Income", min_value=5000, max_value=1000000, value=50000)
pitch_score = st.number_input("Pitch Satisfaction Score", min_value=0, max_value=10, value=5)
product_pitched = st.text_input("Product Pitched", "Wellness")
num_followups = st.number_input("Number of Followups", min_value=0, max_value=20, value=0)
duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=60, value=10)

# -------- CREATE INPUT DF --------
input_data = pd.DataFrame([[
    age, typeofcontact, citytier, occupation, gender, num_persons, preferred_star,
    marital_status, num_trips, passport, own_car, num_children, designation,
    monthly_income, pitch_score, product_pitched, num_followups, duration_pitch
]], columns=[
    "Age","TypeofContact","CityTier","Occupation","Gender","NumberOfPersonVisiting",
    "PreferredPropertyStar","MaritalStatus","NumberOfTrips","Passport","OwnCar",
    "NumberOfChildrenVisiting","Designation","MonthlyIncome","PitchSatisfactionScore",
    "ProductPitched","NumberOfFollowups","DurationOfPitch"
])

# -------- PREDICTION --------
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Purchase: {'Yes' if prediction==1 else 'No'}")
