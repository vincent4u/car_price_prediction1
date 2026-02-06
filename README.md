# Laptop Price Prediction

![Python](https://img.shields.io/badge/python-3.11-blue)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

````markdown
# Car Price Prediction

This project uses machine learning techniques to predict car prices based on various features such as brand, model, year, mileage, and other attributes. It includes exploratory data analysis (EDA), data preprocessing, and model-building notebooks, along with a sample database and dependencies for reproducibility.

## Features

- Data preprocessing and cleaning pipelines
- Exploratory Data Analysis (EDA) notebooks
- Machine learning model implementation for price prediction
- Structured project layout for reproducibility and scalability

## Quick Setup

1. Clone the repository:

```bash
git clone https://github.com/vincent4u/car_price_prediction1.git
````

2. Navigate to the project folder:

```bash
cd car_price_prediction1
```

3. Create and activate a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```
car_price_prediction1/
├── notebooks/            # Jupyter notebooks for EDA and modeling
├── src/                  # Python scripts or modules (optional)
├── data_sample/          # Small sample datasets (optional)
├── models/               # Trained models (optional)
├── requirements.txt      # Project dependencies
├── .gitignore
└── README.md
```

## Notes

* `.env` and large data files (`data/`, `db/`) are ignored and not included in the repository.
* Jupyter notebooks contain the step-by-step workflow for data exploration, preprocessing, and model building.


````markdown
## 🚀 How to Run the Project

From the **project root directory**, run the following commands in your terminal:

### 1️⃣ Train the model
```bash
python -m src.models.train
````

* Trains the model on the dataset
* Saves the trained pipeline to `outputs/trained_model.pkl`

### 2️⃣ Evaluate model + graphs

```bash
python -m src.models.evaluate
```

* Computes metrics (R², RMSE, MAE)
* Creates plots: `actual_vs_predicted.png`, `residuals_hist.png`, `r2_vs_perfect.png`
* Saves metrics CSV: `evaluation_metrics.csv`

### 3️⃣ Feature importance

```bash
python -m src.models.interpret
```

* Extracts feature importances from the trained model
* Saves CSV and plot: `feature_importances.csv`, `feature_importances.png`

### 4️⃣ Predict on new data

```bash
python -m src.models.predict
```

* Makes predictions on new/unseen data in `data/clean_data/new_data.csv`
* Saves predictions to `outputs/predictions.csv`

```
