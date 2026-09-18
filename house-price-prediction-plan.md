# House Price Prediction — Project Plan

## Top-Level Overview

Build a fully self-contained Python project for house price prediction using the provided
`house_price_regression_dataset.csv` (7 numeric features → 1 continuous target `House_Price`).

**Stack**
- **Backend / ML**: scikit-learn (Linear Regression, Random Forest), XGBoost, joblib for model persistence
- **Frontend**: Streamlit — single-page interactive app
- **Data / Viz**: pandas, matplotlib, seaborn, plotly

**Scope**
1. Explore and preprocess the dataset
2. Train, evaluate, and persist three models (Linear Regression, Random Forest, XGBoost)
3. Build a Streamlit UI with a prediction form, model selector, and comparison dashboard

**Non-Goals**
- No database or API server; everything runs locally via `streamlit run`
- No user authentication
- No deployment to cloud

---

## Project Structure (target)

```
house_price_project/
├── data/
│   └── house_price_regression_dataset.csv   # symlink / copy of source CSV
├── models/
│   └── (generated) lr_model.pkl, rf_model.pkl, xgb_model.pkl, best_model.txt
├── notebooks/
│   └── eda.ipynb                            # optional EDA notebook
├── src/
│   ├── data_loader.py                       # load & validate CSV
│   ├── preprocessing.py                     # feature engineering, train/test split, scaling
│   ├── train.py                             # train all three models, evaluate, save
│   └── predict.py                           # load saved model, run inference
├── app.py                                   # Streamlit entry point
├── requirements.txt
└── README.md
```

---

## Sub-Tasks

---

### Sub-Task 1 — Project Scaffold & Dependencies

**Intent**
Create the folder structure, `requirements.txt`, and `README.md` so every subsequent
sub-task has a consistent home.

**Expected Outcomes**
- All directories exist
- `requirements.txt` lists every needed package with pinned major versions
- `README.md` explains how to install and run the app

**Todo List**
1. Create directory tree: `house_price_project/`, `data/`, `models/`, `src/`
2. Copy `house_price_regression_dataset.csv` into `data/`
3. Write `requirements.txt` with: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `joblib`, `matplotlib`, `seaborn`, `plotly`
4. Write `README.md` with setup and run instructions (`pip install -r requirements.txt` → `streamlit run app.py`)

**Relevant Context**
- Source CSV: `house_price_regression_dataset.csv` (root of workspace)
- Features: `Square_Footage`, `Num_Bedrooms`, `Num_Bathrooms`, `Year_Built`, `Lot_Size`, `Garage_Size`, `Neighborhood_Quality`
- Target: `House_Price`

**Status**: `[ ] pending`

---

### Sub-Task 2 — Data Loader & Preprocessing Module

**Intent**
Write `src/data_loader.py` and `src/preprocessing.py` to cleanly load, validate, and
prepare the dataset for training and inference.

**Expected Outcomes**
- `data_loader.load_data(path)` returns a validated pandas DataFrame
- `preprocessing.prepare_data(df)` returns `X_train, X_test, y_train, y_test, scaler` with features scaled (StandardScaler) and an 80/20 stratified-friendly split
- The same `scaler` is used at inference time (saved alongside models)

**Todo List**
1. Write `data_loader.py`:
   - Read CSV with pandas
   - Assert required columns exist
   - Drop rows with nulls (if any)
   - Return clean DataFrame
2. Write `preprocessing.py`:
   - Define `FEATURE_COLS` constant list
   - Split into X (features) and y (target)
   - `train_test_split` with `random_state=42`, `test_size=0.2`
   - Fit `StandardScaler` on training set only
   - Return train/test splits and fitted scaler

**Relevant Context**
- All 7 feature columns are numeric — no categorical encoding needed
- `Lot_Size` is a float; all others are int/float
- No nulls observed in sample rows, but a null-drop guard is good practice

**Status**: `[ ] pending`

---

### Sub-Task 3 — Model Training, Evaluation & Persistence

**Intent**
Write `src/train.py` to train all three models, compute evaluation metrics, persist
every model and the scaler to `models/`, and record which model performed best.

**Expected Outcomes**
- Three `.pkl` files saved: `lr_model.pkl`, `rf_model.pkl`, `xgb_model.pkl`
- `scaler.pkl` saved in `models/`
- `models/best_model.txt` contains the name of the best model (lowest RMSE on test set)
- Console output shows a comparison table: model name | R² | MAE | RMSE

**Todo List**
1. Import data loader and preprocessing helpers
2. Instantiate models:
   - `LinearRegression()`
   - `RandomForestRegressor(n_estimators=100, random_state=42)`
   - `XGBRegressor(n_estimators=100, random_state=42, verbosity=0)`
3. Train each model on scaled `X_train` / `y_train`
4. Evaluate each on `X_test` / `y_test`: compute R², MAE, RMSE
5. Print comparison table
6. Save each model with `joblib.dump` to `models/`
7. Save `scaler` with `joblib.dump` to `models/scaler.pkl`
8. Write name of best-RMSE model to `models/best_model.txt`
9. Ensure `train.py` is runnable as `python src/train.py` from project root

**Relevant Context**
- `src/data_loader.py` and `src/preprocessing.py` from Sub-Task 2
- Models directory: `house_price_project/models/`

**Status**: `[ ] pending`

---

### Sub-Task 4 — Inference Helper

**Intent**
Write `src/predict.py` exposing a clean `predict(features_dict, model_name)` function
that the Streamlit app calls without knowing about file paths or scaling.

**Expected Outcomes**
- `predict.load_model(model_name)` loads the requested `.pkl` from `models/`
- `predict.predict(features_dict, model_name)` returns a single float price prediction
- `predict.get_best_model_name()` reads `models/best_model.txt` and returns the string

**Todo List**
1. Write `load_model(model_name)` — maps name string to file path and calls `joblib.load`
2. Write `load_scaler()` — loads `models/scaler.pkl`
3. Write `predict(features_dict, model_name)`:
   - Accept a dict of feature name → value
   - Convert to DataFrame with correct column order (`FEATURE_COLS`)
   - Scale with loaded scaler
   - Return `model.predict(X)[0]`
4. Write `get_best_model_name()` — reads first line of `models/best_model.txt`

**Relevant Context**
- `FEATURE_COLS` constant defined in `src/preprocessing.py` — import and reuse it
- Models saved in Sub-Task 3

**Status**: `[ ] pending`

---

### Sub-Task 5 — Streamlit Frontend

**Intent**
Build `app.py` — the single Streamlit page with three interactive sections:
(1) Prediction Form, (2) Model Comparison Dashboard, (3) Data Explorer.

**Expected Outcomes**
- Sidebar lets user pick a model (default = best model auto-selected)
- Input form with sliders/number inputs for all 7 features
- "Predict" button shows the estimated price with a styled metric card
- Model comparison tab: bar chart of R², MAE, RMSE for all three models
- Data Explorer tab: scatter plots (Square_Footage vs Price, coloured by Neighborhood_Quality), correlation heatmap
- App runs with `streamlit run app.py` from `house_price_project/`

**Todo List**
1. Import `src/predict.py` helpers; call `get_best_model_name()` for default selection
2. Build sidebar:
   - Model selector (`st.selectbox`) pre-filled with best model name
3. Build "Predict" tab:
   - `st.slider` / `st.number_input` for each of the 7 features with sensible min/max/default
   - "Predict Price" button → call `predict.predict()` → display `st.metric`
4. Build "Model Comparison" tab:
   - Load pre-computed metrics from a `models/metrics.json` file (saved during training)
   - Render plotly bar charts for R², MAE, RMSE side-by-side
5. Build "Data Explorer" tab:
   - Load raw CSV via `data_loader.load_data()`
   - Plotly scatter: `Square_Footage` vs `House_Price`, colour = `Neighborhood_Quality`
   - Seaborn/matplotlib correlation heatmap via `st.pyplot`
6. Add page config: title, icon, wide layout
7. Handle the case where models are not yet trained (show a warning with instructions)

**Relevant Context**
- Sub-Task 3 must add saving of `models/metrics.json` (dict of model_name → {r2, mae, rmse})
- Sub-Task 4 `predict()` function is the only inference surface used here

**Status**: `[ ] pending`

---

### Sub-Task 6 — Integration & Smoke Test

**Intent**
Verify the full pipeline end-to-end: train → save → app loads models → predictions work.

**Expected Outcomes**
- `python src/train.py` completes without errors, all `.pkl` and `.json` files exist
- `streamlit run app.py` opens without import errors
- A sample prediction returns a plausible price (within dataset range ~$150k–$1.2M)
- README accurately reflects the run steps

**Todo List**
1. Run `python src/train.py` and confirm output files in `models/`
2. Run `streamlit run app.py` and verify all three tabs load
3. Enter sample input values and confirm prediction renders
4. Fix any import path issues (`sys.path` adjustments if needed in `app.py`)
5. Update README with any corrections discovered during smoke test

**Relevant Context**
- All prior sub-tasks must be complete before this one
- Working directory assumed to be `house_price_project/` when running the app

**Status**: `[ ] pending`
