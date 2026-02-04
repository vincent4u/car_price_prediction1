import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.ensemble import RandomForestRegressor

from src.features.build_features import build_preprocessor

file_path = "C:/Users/vince/Data_science_projects/car_price_prediction1/data/clean_data/laptop_pricing_cleaned_data.csv"

def evaluate_model(data_path=file_path, cv=5, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    y = df["Price"]
    X = df.drop(columns=["Price"])

    pipeline = Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            ("model", RandomForestRegressor(
                n_estimators=200,
                random_state=42
            ))
        ]
    )

    # Out-of-fold predictions
    y_pred = cross_val_predict(
        pipeline,
        X,
        y,
        cv=5
    )

    # Metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    print(f"R²: {r2:.3f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    # Save metrics
    metrics_df = pd.DataFrame({
        "R2": [r2],
        "RMSE": [rmse],
        "MAE": [mae]
    })
    metrics_df.to_csv(f"{output_dir}/evaluation_metrics.csv", index=False)

    # Plot: Actual vs Predicted
    plt.figure()
    plt.scatter(y, y_pred, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], linestyle="--")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted (CV)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/actual_vs_predicted.png")
    plt.close()

    # Plot: Residuals
    residuals = y - y_pred
    plt.figure()
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual")
    plt.title("Residuals Distribution (CV)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/residuals_hist.png")
    plt.close()

    print(f"Evaluation results saved in '{output_dir}/'")

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae
    }

if __name__ == "__main__":
    evaluate_model()


# """
# Evaluate regression model and save results + graphs.
# """
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# from src.models.persist import load_model

# OUTPUT_DIR = "outputs"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Load trained pipeline
# pipeline = load_model("outputs/trained_model.pkl")

# # Load dataset
# data_path = "data/clean_data/laptop_pricing_cleaned_data.csv"
# df = pd.read_csv(data_path)
# X = df.drop(columns=["Price"])
# y = df["Price"]

# # Make predictions
# y_pred = pipeline.predict(X)

# # Compute metrics
# r2 = r2_score(y, y_pred)
# rmse = np.sqrt(mean_squared_error(y, y_pred))
# mae = mean_absolute_error(y, y_pred)

# # Print metrics
# print(f"R²: {r2:.3f} (closer to 1 is better)")
# print(f"RMSE: {rmse:.2f}")
# print(f"MAE: {mae:.2f}")

# # Save metrics
# metrics_df = pd.DataFrame({
#     "R2": [r2],
#     "RMSE": [rmse],
#     "MAE": [mae]
# })
# metrics_df.to_csv(f"{OUTPUT_DIR}/evaluation_metrics.csv", index=False)

# # Plot actual vs predicted
# plt.figure(figsize=(6, 6))
# plt.scatter(y, y_pred, alpha=0.6)
# plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label="Perfect Prediction")
# plt.xlabel("Actual Price")
# plt.ylabel("Predicted Price")
# plt.title("Actual vs Predicted Prices")
# plt.legend()
# plt.tight_layout()
# plt.savefig(f"{OUTPUT_DIR}/actual_vs_predicted.png")
# plt.close()

# # Plot R² vs 1
# plt.figure(figsize=(6, 4))
# plt.bar(["R²", "Perfect"], [r2, 1], color=["blue", "green"])
# plt.title("R² Compared to Perfect Model")
# plt.ylim(0, 1.05)
# plt.ylabel("R² Value")
# plt.tight_layout()
# plt.savefig(f"{OUTPUT_DIR}/r2_vs_perfect.png")
# plt.close()

# # Residuals histogram
# residuals = y - y_pred
# plt.figure(figsize=(6, 4))
# plt.hist(residuals, bins=30, alpha=0.7)
# plt.title("Residuals Distribution")
# plt.xlabel("Residual")
# plt.ylabel("Count")
# plt.tight_layout()
# plt.savefig(f"{OUTPUT_DIR}/residuals_hist.png")
# plt.close()

# print(f"Evaluation results and plots saved in '{OUTPUT_DIR}/'")
