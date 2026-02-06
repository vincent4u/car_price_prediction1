from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

NUMERIC_COLS = [
    "Category",
    "GPU",
    "OS",
    "CPU_core",
    "Screen_Size_inch",
    "CPU_frequency",
    "RAM_GB",
    "Storage_GB_SSD",
    "Weight_pounds",
    "Full HD",
    "Screen-IPS_panel",
]

def build_preprocessor():
    numeric_transformer = StandardScaler()

    # categorical_transformer = OneHotEncoder(
    #     handle_unknown="ignore"
    

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLS)
            # ("cat", categorical_transformer, CATEGORICAL_COLS)
        ], remainder="drop"
    )

    return preprocessor
