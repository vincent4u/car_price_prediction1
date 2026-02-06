import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from src.features.build_features import build_preprocessor

def test_training_pipeline_runs():
    df = pd.DataFrame({
        "Category": [3, 4, 4],
        "GPU": [1, 2, 2],
        "OS": [1, 1, 1],
        "CPU_core": [3, 7, 5],
        "Screen_Size_inch": [39.6, 39.6, 33.8],
        "CPU_frequency": [0.68, 0.93, 0.55],
        "RAM_GB": [4, 8, 8],
        "Storage_GB_SSD": [256, 256, 128],
        "Weight_pounds": [4.85, 4.85, 2.69],
        "Full HD": [1, 1, 0],
        "Screen-IPS_panel": [0, 0, 1],
        "Price": [634, 946, 1244],
    })

    X = df.drop(columns=["Price"])
    y = df["Price"]

    pipe = Pipeline([
        ("preprocessing", build_preprocessor()),
        ("model", Ridge())
    ])

    pipe.fit(X, y)
    preds = pipe.predict(X)

    assert len(preds) == len(y)