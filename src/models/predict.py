import os
import pandas as pd
from src.models.persist import load_model
from src.features.build_features import preprocess_features

# PROJECT ROOT (one level above src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(PROJECT_ROOT, "outputs", "trained_model.pkl")
new_data_path = os.path.join(PROJECT_ROOT, "data", "clean_data", "new_data.csv")
output_path = os.path.join(PROJECT_ROOT, "outputs", "predictions.csv")

def predict_new_data(model_path, new_data_path, output_path="outputs/predictions.csv"):
    # Load new dataset
    df_new = pd.read_csv(new_data_path)

    # Preprocess features
    X_new = preprocess_features(df_new)

    # Load trained model
    pipeline = load_model(model_path)

    # Predict
    df_new["Predictions"] = pipeline.predict(X_new)

    # Save predictions
    df_new.to_csv(output_path, index=False)
    print(f"Predictions saved at {output_path}")

if __name__ == "__main__":
    predict_new_data(model_path=model_path, new_data_path=new_data_path, output_path=output_path)
