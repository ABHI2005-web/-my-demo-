"""
train_model.py
Generates a realistic synthetic used-car dataset and trains a
RandomForestRegressor to predict car selling price.
Run this once to produce model/car_price_model.pkl
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import pickle
import os

np.random.seed(42)

N = 3000

car_names = [
    "Maruti Swift", "Hyundai i20", "Honda City", "Toyota Innova",
    "Maruti Baleno", "Hyundai Creta", "Tata Nexon", "Mahindra XUV500",
    "Honda Amaze", "Ford EcoSport", "Maruti Alto", "Hyundai Verna",
    "Toyota Fortuner", "Kia Seltos", "Skoda Rapid"
]

fuel_types = ["Petrol", "Diesel", "CNG"]
seller_types = ["Dealer", "Individual"]
transmissions = ["Manual", "Automatic"]

rows = []
current_year = 2026

for _ in range(N):
    car_name = np.random.choice(car_names)
    year = np.random.randint(2005, current_year)
    age = current_year - year

    present_price = round(np.random.uniform(3.5, 35.0), 2)  # in lakhs
    kms_driven = int(np.random.uniform(500, 150000))
    fuel_type = np.random.choice(fuel_types, p=[0.55, 0.35, 0.10])
    seller_type = np.random.choice(seller_types, p=[0.65, 0.35])
    transmission = np.random.choice(transmissions, p=[0.75, 0.25])
    owner = np.random.choice([0, 1, 2, 3], p=[0.7, 0.2, 0.08, 0.02])

    # Build a realistic depreciation-based selling price
    depreciation = 0.92 ** age
    kms_penalty = 1 - min(kms_driven / 300000, 0.35)
    fuel_factor = {"Diesel": 1.05, "Petrol": 1.0, "CNG": 0.95}[fuel_type]
    trans_factor = 1.08 if transmission == "Automatic" else 1.0
    seller_factor = 0.95 if seller_type == "Individual" else 1.0
    owner_factor = 1 - (owner * 0.06)

    noise = np.random.normal(1.0, 0.05)

    selling_price = (
        present_price
        * depreciation
        * kms_penalty
        * fuel_factor
        * trans_factor
        * seller_factor
        * owner_factor
        * noise
    )
    selling_price = max(round(selling_price, 2), 0.3)

    rows.append([
        car_name, year, selling_price, present_price, kms_driven,
        fuel_type, seller_type, transmission, owner
    ])

df = pd.DataFrame(rows, columns=[
    "Car_Name", "Year", "Selling_Price", "Present_Price", "Kms_Driven",
    "Fuel_Type", "Seller_Type", "Transmission", "Owner"
])

# Feature engineering
df["Car_Age"] = current_year - df["Year"]

le_fuel = LabelEncoder()
le_seller = LabelEncoder()
le_trans = LabelEncoder()

df["Fuel_Type_enc"] = le_fuel.fit_transform(df["Fuel_Type"])
df["Seller_Type_enc"] = le_seller.fit_transform(df["Seller_Type"])
df["Transmission_enc"] = le_trans.fit_transform(df["Transmission"])

feature_cols = [
    "Present_Price", "Kms_Driven", "Car_Age",
    "Fuel_Type_enc", "Seller_Type_enc", "Transmission_enc", "Owner"
]

X = df[feature_cols]
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)
print(f"R2 Score: {r2:.4f}")
print(f"MAE: {mae:.4f} lakhs")

os.makedirs("model", exist_ok=True)

with open("model/car_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/encoders.pkl", "wb") as f:
    pickle.dump({
        "fuel": le_fuel,
        "seller": le_seller,
        "transmission": le_trans,
        "current_year": current_year,
        "feature_cols": feature_cols
    }, f)

print("Model and encoders saved to /model")
