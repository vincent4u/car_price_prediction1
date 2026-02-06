---

# Laptop Price Prediction
![Python](https://img.shields.io/badge/python-3.11-blue)
![CI](https://github.com/vincent4u/car_price_prediction1/actions/workflows/ci.yml/badge.svg?branch=main)
![License](https://img.shields.io/badge/license-MIT-blue)


---

## **Project Overview**

This project uses **machine learning pipelines** to predict **claptop prices** based on hardware and specification features such as RAM, CPU frequency and cores, GPU, storage type and size, operating system, screen size, display type, and brand

It includes:

* **Data preprocessing pipelines** for numeric and categorical features
* **Exploratory Data Analysis (EDA)** notebooks
* **Model training, evaluation, and interpretation**
* Structured outputs for **reproducibility**

The project is **modular and industry-ready**, suitable for portfolio presentation.

---

## **Features**

* Automatic **data preprocessing and cleaning**
* Step-by-step **EDA and model-building notebooks**
* **Machine learning model training** using pipelines
* **Evaluation metrics**: R², RMSE, MAE
* **Feature importance extraction**
* Modular project layout for **scalability and reproducibility**

---

## **Project Structure**

```
car_price_prediction1/
├── data/                  # Raw and cleaned datasets
│   ├── clean_data/        # Processed datasets for training/prediction
│   └── new_data/          # Data for inference
├── notebooks/             # EDA and prototyping notebooks
├── outputs/               # Saved models, metrics, and plots
│   ├── models/            # Trained pipelines/models
│   ├── metrics/           # Evaluation metrics CSV
│   └── plots/             # Evaluation and interpretation graphs
├── src/                   # Python modules (features, models, utils)
├── requirements.txt       # Project dependencies
├── .gitignore
└── README.md
```

> **Note:** `.env` and large data files are ignored in the repository.

---

## **Quick Setup**

1. **Clone the repository**

```bash
git clone https://github.com/vincent4u/car_price_prediction1.git
cd car_price_prediction1
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## **How to Run the Project**

Run the scripts from the **project root directory**:

### 1️⃣ Train the model

```bash
python -m src.models.train
```

* Trains the ML pipeline on the dataset
* Saves the trained model to `outputs/models/trained_model.pkl`

---

### 2️⃣ Evaluate model + graphs

```bash
python -m src.models.evaluate
```

* Computes R², RMSE, MAE
* Generates evaluation plots:

  * `actual_vs_predicted.png`
  * `residuals_hist.png`
  * `r2_vs_perfect.png`
* Saves metrics CSV: `evaluation_metrics.csv`

---

### 3️⃣ Feature importance

```bash
python -m src.models.interpret
```

* Extracts feature importances from the trained model
* Saves CSV and plot: `feature_importances.csv` & `feature_importances.png`

---

### 4️⃣ Predict on new data

```bash
python -m src.models.predict
```

* Runs inference on new/unseen data (`data/clean_data/new_data.csv`)
* Saves predictions to `outputs/predictions.csv`

---

