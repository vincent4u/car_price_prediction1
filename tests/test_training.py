import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from src.features.build_features import build_preprocessor

def test_training_pipeline_runs():
    df= pd.DataFrame({"RAM_GB": [8,16],
    "CPU_frequency": [256, 464],
    "CPU_core": [3,5],
    "GPU":[1,3],
    "Storage_GB_SSD": [256, 128],
    "OS": ["Dell","HP"],
    "Category": [3,4],
    "price": [700, 1200]
        })
    
    X = df.drop(columns=["price"])
    y = df["price"]

    pipe = Pipeline(steps=[
        ("preprocessing", build_preprocessor()),
        ("model", Ridge())
    ])
    pipe.fit(X, y)
    preds= pipe.predict(X)

    assert len(preds) == len(y)