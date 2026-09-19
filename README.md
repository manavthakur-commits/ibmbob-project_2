# 🏡House Price Prediction

A fully self-contained Python project that trains three regression models on a house price dataset and serves an interactive Streamlit web app for predictions, model comparison, and data exploration.

Dataset: `house_price_regression_dataset.csv` | 1,000 records | 7 features
Stack: Python | scikit-learn | XGBoost | Streamlit

---

## Project Structure

```
house_price_project/
├── data/
│   └── house_price_regression_dataset.csv
├── models/          # populated after running train.py
│   ├── lr_model.pkl
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   ├── best_model.txt
│   └── metrics.json
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
├── app.py
├── requirements.txt
└── README.md
```

### Module responsibilities

| Module | Responsibility |
|---|---|
| `src/data_loader.py` | Load CSV, validate required columns, drop nulls |
| `src/preprocessing.py` | Define feature columns, train/test split, `StandardScaler` |
| `src/train.py` | Train all 3 models, compute metrics, persist `.pkl` files and `metrics.json` |
| `src/predict.py` | Load saved models, run scaled inference, cache models in-process |
| `app.py` | Streamlit UI: Predict tab, Model Comparison tab, Data Explorer tab |

**Data flow:** CSV → `data_loader` → `preprocessing` (scale) → `train.py` (fit + save) → `models/*.pkl` → `predict.py` → Streamlit UI

---

## Features

| Feature | Description |
|---|---|
| **3 Models** | Linear Regression, Random Forest, XGBoost |
| **Auto-select best** | Best model (lowest RMSE) selected by default in the UI |
| **Prediction Form** | Sliders & inputs for all 7 house features |
| **Model Comparison** | R², MAE, RMSE charts for all three models |
| **Data Explorer** | Scatter plots and correlation heatmap |

### Key objectives

- Train and evaluate three regression models: Linear Regression, Random Forest, and XGBoost.
- Automatically select and persist the best-performing model based on RMSE.
- Provide an interactive Streamlit UI for real-time predictions, model comparison, and data exploration.
- Keep the project fully local with no external APIs or databases required.

### Technology stack

| Component | Technology / Library |
|---|---|
| ML Backend | scikit-learn, XGBoost, joblib |
| Data Processing | pandas, numpy |
| Frontend UI | Streamlit (Python) |
| Visualisation | Plotly, Matplotlib, Seaborn |
| Language / Version | Python 3.13 |

---

## Setup

### 1. Install dependencies

```bash
cd house_price_project
pip install -r requirements.txt
```

### 2. Train models

```bash
python src/train.py
```

This trains all three models on an 80/20 train/test split (`random_state=42`), then prints a metrics comparison table and saves `lr_model.pkl`, `rf_model.pkl`, `xgb_model.pkl`, `scaler.pkl`, `metrics.json`, and `best_model.txt` into `models/`.

### 3. Run the app

```bash
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. The app is ready to use immediately.

---

## Dataset Features

The dataset contains 1,000 residential property records with 7 numeric features and one continuous target (`House_Price`). All features are numeric — no categorical encoding is required.

| Column | Type | Description |
|---|---|---|
| `Square_Footage` | int | Total living area in square feet (400–6,000) |
| `Num_Bedrooms` | int | Number of bedrooms (1–6) |
| `Num_Bathrooms` | int | Number of bathrooms (1–5) |
| `Year_Built` | int | Year of construction (1950–2024) |
| `Lot_Size` | float | Lot area in acres (0.1–10.0) |
| `Garage_Size` | int | Garage capacity (0, 1, or 2 cars) |
| `Neighborhood_Quality` | int | Quality score 1–10 (higher = better) |
| `House_Price` | float | **Target** — sale price in USD (approx. $150,000–$1,200,000) |

### Preprocessing steps

- Null guard: rows with missing values are dropped (none found in this dataset).
- 80/20 train/test split with `random_state=42` for reproducibility.
- `StandardScaler` fitted on the training set only; applied to the test set at evaluation and to new input at inference time.

---

## Machine Learning Models

All three models were trained on 800 samples (80% of 1,000) and evaluated on the held-out 200-sample test set.

- **Linear Regression** — classical ordinary least-squares baseline, no hyperparameters tuned. Achieved the best performance on this dataset, indicating the feature-to-price relationship is predominantly linear.
- **Random Forest Regressor** — ensemble of 100 decision trees (`n_estimators=100, random_state=42, n_jobs=-1`). Handles non-linearities and feature interactions automatically but showed more variance than the linear model here.
- **XGBoost Regressor** — gradient-boosted trees with 100 estimators (`n_estimators=100, random_state=42`). Achieved the second-best RMSE, just behind Linear Regression and ahead of Random Forest.

### Model evaluation results

| Model | R² Score | MAE (USD) | RMSE (USD) |
|---|---|---|---|
| **Linear Regression (BEST)** | 0.9984 | $8,175 | $10,071 |
| XGBoost | 0.9952 | $14,254 | $17,522 |
| Random Forest | 0.9939 | $16,129 | $19,874 |

The best model is auto-saved to `models/best_model.txt`. All three models achieved R² > 0.99, confirming an excellent fit; Linear Regression outperformed both ensemble methods, consistent with a strong linear relationship between square footage and price.

**Sample prediction** (2,000 sq ft | 3 bed | 2 bath | built 2000 | 2.0 acres | Garage 1 | Neighborhood Quality 7):

| Model | Predicted Price |
|---|---|
| Linear Regression | $458,348 |
| Random Forest | $456,937 |
| XGBoost | $474,264 |

---

## Streamlit App Walkthrough

The app has a sidebar plus three tabs.

**Sidebar — Model Selector**
App title, a model selector dropdown, and a label showing the auto-selected best model. The default selection reads from `models/best_model.txt` automatically — no user action required.

**Tab 1 — Predict Price**
Two columns of input controls: Square Footage, Bedrooms, Bathrooms, and Year Built sliders on the left; Lot Size, Garage Size, and Neighborhood Quality on the right. Clicking **Predict Price** sends all 7 inputs to the selected model and displays the result as a large metric card (e.g. $458,348), with an expandable Input Summary table.

**Tab 2 — Model Comparison**
Shows all three models' metrics side-by-side (loaded from the pre-computed `models/metrics.json`, no re-training): a metrics table plus three Plotly bar charts (R², MAE, RMSE) and a highlighted info box naming the best model.

**Tab 3 — Data Explorer**
Exploratory data analysis of the raw dataset:
1. Raw data preview (first 100 rows) in a collapsible expander
2. Scatter plot: Square Footage vs House Price, colour-coded by Neighborhood Quality
3. Histogram: House Price distribution
4. Correlation heatmap (lower triangle, all 8 columns)
5. Box plot: House Price by Number of Bedrooms
6. Box plot: House Price by Neighborhood Quality

---

## Project Files Reference

| File | Purpose |
|---|---|
| `data/house_price_regression_dataset.csv` | Raw dataset (1,000 rows, 8 columns) |
| `src/data_loader.py` | CSV loading and validation |
| `src/preprocessing.py` | Feature list, train/test split, `StandardScaler` |
| `src/train.py` | Model training, evaluation, and persistence |
| `src/predict.py` | Inference helper with in-process model cache |
| `app.py` | Streamlit application (3 tabs) |
| `models/metrics.json` | Pre-computed evaluation metrics for all 3 models |
| `models/best_model.txt` | Name of the best model (Linear Regression) |
| `requirements.txt` | Python package dependencies |

---

## Conclusion

This project demonstrates a complete machine learning pipeline from raw data to an interactive UI, built entirely in Python.

- All three models achieved R² > 0.99, confirming the seven features are highly predictive of house price.
- Linear Regression achieved the best results (R² = 0.9984, RMSE = $10,071), outperforming both ensemble methods — consistent with the dataset's strong linear relationships.
- The Streamlit frontend provides a clean, usable interface requiring zero web development knowledge.
- The modular architecture makes the project easy to extend with new models, features, or datasets.
