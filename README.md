# 🏦 Credit Risk & Scoring Engine

Modular loan risk assessment web app: an XGBoost model predicts default probability, a rule engine
produces explainable improvement advice, and a Streamlit dashboard shows the result with a Plotly gauge.

## Features
- **ML pipeline:** `scikit-learn` `Pipeline` + `ColumnTransformer` (scaling, one-hot encoding, derived ratio features) with XGBoost.
- **Imbalanced data:** stratified split and `scale_pos_weight`, evaluated with ROC-AUC and a classification report.
- **Explainability:** rule-based recommendations (`src/recommendations.py`) covering debt-to-income, loan-to-income, employment length, savings, and more.
- **Dashboard:** sidebar inputs, approval-probability gauge (0-100%), and a colour-coded risk band (green / yellow / red).

## Project Structure
```
credit-risk-engine/
├── data/german_credit_data.csv
├── models/{train_model.py, credit_model.pkl}
├── src/{__init__.py, preprocessing.py, recommendations.py}
├── app.py
├── requirements.txt
└── README.md
```

## Quick Start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python models/train_model.py
streamlit run app.py
```

## Data Note
The public German Credit dataset has no income / debt-to-income fields or a default label.
If `data/german_credit_data.csv` does not follow the expected schema
(`age, monthly_income, loan_amount, loan_term_months, debt_to_income, employment_years, housing, savings_level, default`),
the training script generates a **synthetic** imbalanced dataset in that schema. Replace it with real data
to use the system beyond a demo.

## Design Principles
- **Single responsibility:** preprocessing, training, rules and UI live in separate modules.
- **Open/closed:** add a new advice rule by appending a `Rule` to `RULES`, with no other code changes.
- **Reproducibility:** fixed seeds, the whole preprocessing stack serialized with the model.

## Roadmap
SHAP-based feature attributions, probability calibration, Docker image, FastAPI scoring endpoint.

## Disclaimer
For educational purposes only. Not financial advice.