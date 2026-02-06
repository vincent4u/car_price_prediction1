import numpy as np
from sklearn.metrics import r2_score
from src.models.evaluate import evaluate_model

def test_r2_perfect_prediction():
    y_true = np.array([100, 200, 300])
    y_pred = np.array([100, 200, 300])

    assert r2_score(y_true, y_pred) == 1.0
