# 1. Install deps
pip install -r requirements.txt

# 2. Train models (generates .joblib files)
python project1_supplier_risk/supplier_risk_model.py

python project2_component_obsolescence/obsolescence_model.py

python project3_demand_forecasting/demand_forecast_model.py

# 3. Start API
cd deployment
python -m uvicorn app:app --host 0.0.0.0 --port 8000
