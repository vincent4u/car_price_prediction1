# src/models/train.py

import pandas as pd
from src.features.build_features import preprocess_features , 
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load clean data
file_path="C:/Users/vince/Data_science_projects/car_price_prediction1/data/clean_data/laptop_pricing_cleaned_data.csv"
df=pd.read_csv(file_path)

# Feature selection and preprocessing
X = preprocess_features(df)
y = df["Price"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# model = LinearRegression()
# model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("R^2 score:", r2_score(y_test, y_pred))
