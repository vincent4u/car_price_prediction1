from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

NUMERIC_COLS = [
    "RAM_GB",
    "CPU_frequency",
    "CPU_core",
    "GPU",
    "Storage_GB_SSD"
]

CATEGORICAL_COLS = [
    "OS",
    "Category"
]

def build_preprocessor():
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS)
        ]
    )

    return preprocessor
