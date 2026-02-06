import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from src.features.build_features import build_preprocessor
from src.models.persist import save_model

file_path = "C:/Users/vince/Data_science_projects/car_price_prediction1/data/clean_data/laptop_pricing_cleaned_data.csv"

def train_model(data_path=file_path):
    df = pd.read_csv(data_path) 
    y = df["Price"]
    X = df.drop(columns=["Price"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


    test_data_folder_path="C:/Users/vince/Data_science_projects/car_price_prediction1/data/test_data"
    os.makedirs(test_data_folder_path, exist_ok=True)
    X_test_file_name = "X_test.csv"
    y_test_file_name = "y_test.csv"
    X_test_full_path = os.path.join(test_data_folder_path,X_test_file_name)
    y_test_full_path = os.path.join(test_data_folder_path,y_test_file_name)
    
    # Save test data
    X_test.to_csv(X_test_full_path, index=False)
    y_test.to_csv(y_test_full_path, index=False)
    
    preprocessor = build_preprocessor()

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        min_samples_split=5,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    save_model(pipeline, model_name="trained_model.pkl")

    return pipeline

if __name__ == "__main__":
    train_model()
