import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="India House Price Predictor 🇮🇳", layout="centered")

st.title("🏠 India House Price Predictor")
st.write("Estimate house prices for *Bangalore, Chennai, Delhi, Hyderabad, or Mumbai* using Machine Learning.")

# ---------- STEP 1: Load data ----------
@st.cache_data
def load_data():
    # You can later replace this part with actual city-wise CSVs
    data = {
        "City": ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Mumbai"] * 100,
        "total_sqft": np.random.randint(500, 5000, 500),
        "bath": np.random.randint(1, 5, 500),
        "bhk": np.random.randint(1, 5, 500),
        "price": np.random.randint(30, 300, 500)  # in lakhs
    }
    return pd.DataFrame(data)

df = load_data()

# ---------- STEP 2: Train model ----------
X = df[["total_sqft", "bath", "bhk"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# ---------- STEP 3: User Inputs ----------
st.subheader("Enter Property Details")

city = st.selectbox("Select City", ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Mumbai"])
sqft = st.number_input("Total Area (sqft)", min_value=300, max_value=10000, value=1000)
bath = st.slider("Number of Bathrooms", 1, 5, 2)
bhk = st.slider("Number of Bedrooms (BHK)", 1, 5, 2)

# ---------- STEP 4: Prediction ----------
if st.button("🔮 Predict Price"):
    prediction = model.predict([[sqft, bath, bhk]])[0]

    # Adding simple city-based multiplier (just for realism)
    city_factor = {
        "Bangalore": 1.0,
        "Chennai": 0.9,
        "Delhi": 1.4,
        "Hyderabad": 1.1,
        "Mumbai": 1.8
    }

    final_price = prediction * city_factor[city]
    st.success(f"🏡 Estimated Price in {city}: ₹{final_price*100000:,.2f}")

st.write("(Demo uses generated sample data. Replace with real datasets for accuracy.)")
