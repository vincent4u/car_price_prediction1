import pandas as pd
import numpy as np
from src.features.build_features import build_preprocessor

def test_preprocessor_no_nan():
    df = pd.DataFrame({
        "Category": [3, 4],
        "GPU": [1, 3],
        "OS": [1, 1],
        "CPU_core": [3, 5],
        "Screen_Size_inch": [39.6, 33.8],
        "CPU_frequency": [0.68, 0.93],
        "RAM_GB": [8, 16],
        "Storage_GB_SSD": [256, 128],
        "Weight_pounds": [4.8, 2.7],
        "Full HD": [1, 0],
        "Screen-IPS_panel": [0, 1],
        "Price": [700, 1200],
    })

    X = df.drop(columns=["Price"])

    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    assert not np.isnan(X_transformed).any()
    assert X_transformed.shape[0] == X.shape[0]
    assert X_transformed.shape[1] == len(X.columns)  # Since all are