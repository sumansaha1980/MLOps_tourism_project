import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="sumansaha1980/Tourism_Package", filename="best_wellness_tourism_model_v1.joblib")
model = joblib.load(model_path)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Wellness Tourism Prediction App")

st.write("""
This application predicts potential buyers of Wellness Tourism Package based on customer data.
Please enter **Customer Data** and **Customer Interaction Data** below to get a prediction.
""")

# ------------------------------
# User Inputs
# ------------------------------
st.subheader("Customer Details")

CustomerID = st.text_input("CustomerID (Unique ID)", value="12345")  # Not used in model but for reference
Age = st.number_input("Age", min_value=0, max_value=120, value=35)
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer", "Others"])
Gender = st.radio("Gender", ["Male", "Female", "Other"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, value=2)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
NumberOfTrips = st.number_input("Average Number of Trips per year", min_value=0, value=2)
Passport = st.radio("Has Passport?", [0, 1])
OwnCar = st.radio("Owns a Car?", [0, 1])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, value=0)
Designation = st.text_input("Designation", value="Manager")
MonthlyIncome = st.number_input("Monthly Income", min_value=0, value=50000)

st.subheader("Customer Interaction Data")

PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
NumberOfFollowups = st.number_input("Number of Followups", min_value=0, value=2)
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0, value=20)

# ------------------------------
# Prepare Input for Prediction
# ------------------------------
input_data = {
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "ProductPitched": ProductPitched,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch
}

input_df = pd.DataFrame([input_data])

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Use custom threshold as dataset is imbalanced on target column
    # where only 19% has taken the product
    classification_threshold = 0.45
    prediction = (probability >= classification_threshold).astype(int)

    if prediction == 1:
        st.success(f"✅ This customer is **likely to purchase** the Wellness Tourism Package. (Confidence: {probability:.2f})")
    else:
        st.error(f"❌ This customer is **unlikely to purchase** the package. (Confidence: {probability:.2f})")
