import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from src.features.build_features import build_preprocessor

file_path = "C:/Users/vince/Data_science_projects/car_price_prediction1/data/clean_data/laptop_pricing_cleaned_data.csv"

def tune_model(data_path=file_path, cv=5, output_dir="outputs/plots", output_dim="outputs/metrics"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dim, exist_ok=True)

    df = pd.read_csv(data_path)
    y = df["Price"]
    X = df.drop(columns=["Price"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    results = {}

    # ---------------- Ridge ----------------
    ridge_pipeline = Pipeline([
        ("preprocessing", build_preprocessor()),
        ("model", Ridge())
    ])
    ridge_params = {"model__alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]}
    ridge_grid = GridSearchCV(
        ridge_pipeline,
        param_grid=ridge_params,
        cv=cv,
        scoring="r2",
        n_jobs=1,
        return_train_score=True
    )
    ridge_grid.fit(X_train, y_train)
    ridge_pred = ridge_grid.predict(X_test)

    results["Ridge"] = {
        **{
            "R2": r2_score(y_test, ridge_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, ridge_pred)),
            "MAE": mean_absolute_error(y_test, ridge_pred)
        },
        "BestParams": ridge_grid.best_params_
    }

    # ---------------- Lasso ----------------
    lasso_pipeline = Pipeline([
        ("preprocessing", build_preprocessor()),
        ("model", Lasso())
    ])
    lasso_params = {"model__alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10]}
    lasso_grid = GridSearchCV(
        lasso_pipeline,
        param_grid=lasso_params,
        cv=cv,
        scoring="r2",
        n_jobs=1,
        return_train_score=True
    )
    lasso_grid.fit(X_train, y_train)
    lasso_pred = lasso_grid.predict(X_test)

    results["Lasso"] = {
        **{
            "R2": r2_score(y_test, lasso_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, lasso_pred)),
            "MAE": mean_absolute_error(y_test, lasso_pred)
        },
        "BestParams": lasso_grid.best_params_
    }

    # ---------------- Linear Regression ----------------
    linear_pipeline = Pipeline([
        ("preprocessing", build_preprocessor()),
        ("model", LinearRegression())
    ])
    linear_pipeline.fit(X_train, y_train)
    linear_pred = linear_pipeline.predict(X_test)

    results["LinearRegression"] = {
        "R2": r2_score(y_test, linear_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, linear_pred)),
        "MAE": mean_absolute_error(y_test, linear_pred),
        "BestParams": None  # no hyperparameters to tune
    }
    # ---------------- Random Forest ----------------
    model = RandomForestRegressor(random_state=42)
    rf_pipeline = Pipeline(steps=[
            ("preprocessing",build_preprocessor()),
            ("model", model)
        ])
    rf_param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5]}

    rf_grid= GridSearchCV(rf_pipeline,
                          param_grid= rf_param_grid,
                          scoring="r2",
                          cv=cv)
    
    rf_grid.fit(X_train,y_train)
    rf_pred = rf_grid.predict(X_test)
    results["RandomForestRegressor"] = {
        "R2": r2_score(y_test, rf_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, rf_pred)),
        "MAE": mean_absolute_error(y_test, rf_pred),
        "BestParams": rf_grid.best_params_  
    }

    # ---------------- Save metrics ----------------
    metrics_df = pd.DataFrame(results).T
    metrics_df.to_csv(f"{output_dim}/evaluation_metrics.csv", index=True)

    # ---------------- Plots ----------------
    for model_name, y_pred in zip(
        ["Ridge", "Lasso", "LinearRegression", "RandomForestRegressor"], 
        [ridge_pred, lasso_pred, linear_pred, rf_pred]
    ):
        # Actual vs Predicted
        plt.figure()
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()],
                 linestyle="--")
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title(f"{model_name}: Actual vs Predicted")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name}_actual_vs_predicted.png")
        plt.close()

        # Residuals
        residuals = y_test - y_pred
        plt.figure()
        plt.hist(residuals, bins=30)
        plt.xlabel("Residual")
        plt.title(f"{model_name}: Residuals Distribution")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name}_residuals_hist.png")
        plt.close()

    print(f"Evaluation complete. Metrics and plots saved in '{output_dir}/'.")
    return results


if __name__ == "__main__":
    res = tune_model()
    print(res)



