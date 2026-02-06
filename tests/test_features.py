import pandas as pd
import numpy as np
from src.features.build_features import build_preprocessor

def test_preprocessor_no_nan():
    df =pd.DataFrame({
        
    "RAM_GB": [8,16],
    "CPU_frequency": [256, 464],
    "CPU_core": [3,5],
    "GPU":[1,3],
    "Storage_GB_SSD": [256, 128],
    "OS": ["Dell","HP"],
    "Category": [3,4],
    "price": [700, 1200]
        })
    
    X =  df.drop(columns=["price"])
    preprocessor = build_preprocessor() 
    X_transformed = preprocessor.fit_transform(X)

    assert not np.isnan(X_transformed).any()