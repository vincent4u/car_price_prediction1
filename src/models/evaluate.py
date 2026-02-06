"""
Evaluate regression model and save results + graphs.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.models.persist import load_model
from sklearn.model_selection import train_test_split

def evaluate_model(data_path, model_path, output_dir="outputs/evaluation"):
    os.makedirs(output_dir, exist_ok=True)

    # Load trained pipeline
    pipeline = load_model(model_path)

    # Load dataset
    # df = pd.read_csv(data_path)
    # X = df.drop(columns=["Price"])
    # y = df["Price"]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load test data
    test_data_folder_path = "C:/Users/vince/Data_science_projects/car_price_prediction1/data/test_data"
    X_test = pd.read_csv(os.path.join(test_data_folder_path, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(test_data_folder_path, "y_test.csv"))["Price"]

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Compute metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Print metrics
    print(f"R²: {r2:.3f} (closer to 1 is better)")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")

    # Save metrics
    metrics_df = pd.DataFrame({
        "R2": [r2],
        "RMSE": [rmse],
        "MAE": [mae]
    })
    metrics_df.to_csv(f"{output_dir}/evaluation_metrics.csv", index=False)

    # Plot actual vs predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Prediction")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted Prices")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/actual_vs_predicted.png")
    plt.close()

    # Plot R² vs 1
    plt.figure(figsize=(6, 4))
    plt.bar(["R²", "Perfect"], [r2, 1], color=["blue", "green"])
    plt.title("R² Compared to Perfect Model")
    plt.ylim(0, 1.05)
    plt.ylabel("R² Value")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/r2_vs_perfect.png")
    plt.close()

    # Residuals histogram
    residuals = y_test - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.title("Residuals Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/residuals_histogram.png")
    plt.close()
    print(f"Evaluation complete. Metrics and plots saved in '{output_dir}/'.")
    return {"R2": r2, "RMSE": rmse, "MAE": mae}

if __name__ == "__main__":
    data_path = "C:/Users/vince/Data_science_projects/car_price_prediction1/data/clean_data/laptop_pricing_cleaned_data.csv"
    model_path = "outputs/models/trained_model.pkl"
    evaluate_model(data_path, model_path, output_dir="outputs/evaluation")
    