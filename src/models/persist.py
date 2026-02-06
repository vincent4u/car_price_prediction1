"""
Save trained models and preprocessing artifacts for future use.
"""

import joblib
import os

def save_model(model, model_name="trained_model.pkl", output_dir="outputs/models"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, model_name)
    joblib.dump(model, path)
    print(f"Model saved at {path}")

def load_model(model_path):
    """Load a persisted model."""
    return joblib.load(model_path)
