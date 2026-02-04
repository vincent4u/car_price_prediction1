import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from src.features.build_features import build_preprocessor
from src.models.persist import save_model

file_path = "C:/Users/vince/Data_science_projects/car_price_prediction1/data/clean_data/laptop_pricing_cleaned_data.csv"

def train_model(data_path=file_path):
    df = pd.read_csv(data_path)

    y = df["Price"]
    X = df.drop(columns=["Price"])

    preprocessor = build_preprocessor()

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X, y)

    save_model(pipeline, model_name="trained_model.pkl")

    return pipeline

if __name__ == "__main__":
    train_model()
