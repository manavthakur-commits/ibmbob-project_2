# House Price Prediction

A fully self-contained Python project that trains three regression models on a house price dataset and serves an interactive Streamlit web app for predictions, model comparison, and data exploration.

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

---

## Features

| Feature | Description |
|---|---|
| **3 Models** | Linear Regression, Random Forest, XGBoost |
| **Auto-select best** | Best model (lowest RMSE) selected by default in the UI |
| **Prediction Form** | Sliders & inputs for all 7 house features |
| **Model Comparison** | R², MAE, RMSE charts for all three models |
| **Data Explorer** | Scatter plots and correlation heatmap |

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

This will print a metrics comparison table and save all model files into `models/`.

### 3. Run the app

```bash
python -m streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Dataset Features

| Column | Type | Description |
|---|---|---|
| `Square_Footage` | int | Total square footage of the house |
| `Num_Bedrooms` | int | Number of bedrooms |
| `Num_Bathrooms` | int | Number of bathrooms |
| `Year_Built` | int | Year the house was built |
| `Lot_Size` | float | Lot size in acres |
| `Garage_Size` | int | Garage capacity (0, 1, or 2 cars) |
| `Neighborhood_Quality` | int | Quality score 1–10 |
| `House_Price` | float | **Target** — sale price in USD |
