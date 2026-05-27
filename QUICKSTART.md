# Quick Start Guide

## Activate Virtual Environment (every time!)

On your Mac, run this in the terminal:

```bash
cd ~/churn-prediction
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal prompt.

## Option 1: Run Jupyter Notebooks

```bash
jupyter notebook
```

This opens a browser window. Navigate to and open notebooks in order:
1. `notebooks/01_eda.ipynb` - See why customers churn
2. `notebooks/02_feature_engineering.ipynb` - Prepare data
3. `notebooks/03_modeling.ipynb` - Compare 3 models
4. `notebooks/04_results.ipynb` - Analyze business impact

## Option 2: Run Interactive Dashboard

```bash
streamlit run app.py
```

This opens an interactive app where you can:
- Input a customer's profile (tenure, charges, contract type)
- See their churn probability
- View feature importance
- See annual savings potential

## Key Commands

```bash
# Activate environment (run this first!)
source venv/bin/activate

# Launch Jupyter
jupyter notebook

# Launch Streamlit
streamlit run app.py

# Run Python scripts
python3 train_models.py

# Deactivate environment (when done)
deactivate
```

## File Locations

- **Raw data:** `data/raw/telco_churn.csv` (7,043 customers)
- **Trained models:** `models/xgboost_churn.pkl`, `models/scaler.pkl`
- **Notebooks:** `notebooks/01_eda.ipynb` through `04_results.ipynb`
- **Python code:** `src/data_loader.py`, `src/models.py`
- **Dashboard:** `app.py`

## What to Explore

1. **Start with EDA** (`01_eda.ipynb`):
   - 50% churn rate in first 6 months
   - Month-to-month contracts = 42% churn
   - Fiber Optic service = 42% churn

2. **See model performance** (`03_modeling.ipynb`):
   - Logistic Regression baseline: 0.842 AUC
   - Random Forest: 0.843 AUC
   - XGBoost champion: 0.841 AUC

3. **Understand business impact** (`04_results.ipynb`):
   - Optimal threshold: 0.42 (not 0.50!)
   - Saves 355 customers/year
   - Protects $1.78M annual value

## Need Help?

Check `README.md` for full documentation!
