import os
import pandas as pd
import matplotlib.pyplot as plt

from src.models.persist import load_model

def extract_feature_importance(
    model_path="outputs/trained_model.pkl",
    output_dir="outputs",
    top_n=20
):
    os.makedirs(output_dir, exist_ok=True)

    pipeline = load_model(model_path)

    preprocessor = pipeline.named_steps["preprocessing"]
    model = pipeline.named_steps["model"]

    # Numeric features
    numeric_features = preprocessor.transformers_[0][2]

    # Categorical features
    categorical_encoder = preprocessor.transformers_[1][1]
    categorical_features = categorical_encoder.get_feature_names_out(
        preprocessor.transformers_[1][2]
    )

    feature_names = list(numeric_features) + list(categorical_features)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    # Save CSV
    importance_df.to_csv(
        f"{output_dir}/feature_importances.csv",
        index=False
    )

    # Plot top N
    top_features = importance_df.head(top_n)

    plt.figure(figsize=(8, 5))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importances.png")
    plt.close()

    print(f"Feature importances saved in '{output_dir}/'")

    return importance_df
