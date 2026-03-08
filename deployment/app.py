"""
Z2Data ML Prediction API - Production Ready
Author: Ahmed Mohammed Gouda
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
import numpy as np
import os
from datetime import datetime

app = FastAPI(
    title="Z2Data ML Prediction API",
    description="Supply Chain ML Models - Supplier Risk, Obsolescence, Demand Forecast",
    version="1.0.0",
)

# ── Find models from project directories ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

models = {}

def try_load(name, paths_dict):
    """Try loading model files from multiple possible locations"""
    result = {}
    for key, possible_paths in paths_dict.items():
        for p in possible_paths:
            if os.path.exists(p):
                result[key] = joblib.load(p)
                print(f"    {key}: {p}")
                break
    if "model" in result:
        models[name] = result
        return True
    return False

print("\n=== Loading ML Models ===")

# Supplier Risk
print("  [1] Supplier Risk:")
try_load("supplier_risk", {
    "model": [
        os.path.join(PARENT_DIR, "project1_supplier_risk", "models", "gradient_boosting_tuned.joblib"),
    ],
    "scaler": [
        os.path.join(PARENT_DIR, "project1_supplier_risk", "models", "scaler.joblib"),
    ],
    "imputer": [
        os.path.join(PARENT_DIR, "project1_supplier_risk", "models", "imputer.joblib"),
    ],
})

# Obsolescence
print("  [2] Obsolescence:")
try_load("obsolescence", {
    "model": [
        os.path.join(PARENT_DIR, "project2_component_obsolescence", "models", "best_model.joblib"),
    ],
    "scaler": [
        os.path.join(PARENT_DIR, "project2_component_obsolescence", "models", "scaler.joblib"),
    ],
    "imputer": [
        os.path.join(PARENT_DIR, "project2_component_obsolescence", "models", "imputer.joblib"),
    ],
})

# Demand
print("  [3] Demand Forecast:")
try_load("demand", {
    "model": [
        os.path.join(PARENT_DIR, "project3_demand_forecasting", "models", "best_forecasting_model.joblib"),
    ],
})

print(f"\n=== Models Ready: {list(models.keys())} ===\n")


# ══════════════════════════════════════════
# Schemas
# ══════════════════════════════════════════

class SupplierRiskRequest(BaseModel):
    years_in_business: int = Field(15, ge=1)
    num_manufacturing_sites: int = Field(3, ge=1)
    annual_revenue_millions: float = Field(500, ge=0)
    num_employees: int = Field(2000, ge=1)
    on_time_delivery_rate: float = Field(0.92, ge=0, le=1)
    defect_rate_ppm: float = Field(300, ge=0)
    lead_time_days: int = Field(21, ge=1)
    lead_time_variability: float = Field(0.12, ge=0, le=1)
    financial_health_score: float = Field(72, ge=0, le=100)
    debt_to_equity_ratio: float = Field(0.8, ge=0)
    num_certifications: int = Field(4, ge=0)
    has_iso9001: int = Field(1, ge=0, le=1)
    has_iso14001: int = Field(1, ge=0, le=1)
    has_iatf16949: int = Field(0, ge=0, le=1)
    num_customers: int = Field(50, ge=1)
    single_source_pct: float = Field(0.2, ge=0, le=1)
    sub_tier_visibility: int = Field(2, ge=0, le=3)
    recent_disruption_events: int = Field(0, ge=0)
    news_sentiment_score: float = Field(0.7, ge=0, le=1)
    compliance_violations: int = Field(0, ge=0)
    geographic_risk_index: float = Field(0.35, ge=0, le=1)
    country: str = Field("Taiwan")
    component_category: str = Field("Semiconductors")

class SupplierRiskResponse(BaseModel):
    risk_label: str
    risk_score: float
    risk_probabilities: dict
    top_risk_factors: List[str]
    recommendation: str

class ObsolescenceRequest(BaseModel):
    years_since_introduction: float = Field(8.0)
    technology_node_nm: int = Field(65)
    lifecycle_stage: str = Field("Mature")
    num_alternative_parts: int = Field(3)
    num_authorized_distributors: int = Field(5)
    monthly_demand_units: int = Field(5000)
    demand_trend_6m: float = Field(-0.05)
    demand_trend_12m: float = Field(-0.1)
    avg_lead_time_weeks: float = Field(6.0)
    lead_time_increase_pct: float = Field(15.0)
    price_trend_6m: float = Field(0.03)
    num_pcn_notices: int = Field(1)
    last_pcn_months_ago: float = Field(6.0)
    manufacturer_financial_health: float = Field(65.0)
    num_design_wins: int = Field(20)
    cross_reference_count: int = Field(3)
    rohs_compliant: int = Field(1, ge=0, le=1)
    automotive_qualified: int = Field(0, ge=0, le=1)
    military_grade: int = Field(0, ge=0, le=1)

class ObsolescenceResponse(BaseModel):
    eol_probability: float
    eol_prediction: str
    risk_level: str
    confidence: str
    recommended_actions: List[str]

class DemandForecastRequest(BaseModel):
    recent_demands: List[float] = Field(default=[5000, 5200, 4800, 5100, 5300, 5500], min_length=6)
    avg_unit_price: float = Field(2.5)
    avg_lead_time_weeks: float = Field(4.0)
    num_orders: int = Field(100)
    market_index: float = Field(105.0)
    month: int = Field(3, ge=1, le=12)
    category: str = Field("Semiconductors")

class DemandForecastResponse(BaseModel):
    forecasted_demand: float
    confidence_interval: dict
    trend: str
    model_used: str


# ══════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════

@app.get("/")
async def root():
    return {"message": "Z2Data ML API", "models": list(models.keys()), "docs": "/docs"}

@app.get("/health")
async def health():
    return {"status": "healthy", "models": list(models.keys()), "time": datetime.now().isoformat()}


@app.post("/predict/supplier-risk", response_model=SupplierRiskResponse)
async def predict_supplier_risk(req: SupplierRiskRequest):
    if "supplier_risk" not in models:
        raise HTTPException(503, "Model not loaded. Run project1 first to generate model files.")
    
    countries_map = {'China': 1, 'Taiwan': 12, 'South Korea': 10, 'Japan': 4, 'Germany': 3,
                     'USA': 13, 'Vietnam': 14, 'India': 5, 'Thailand': 11, 'Mexico': 7,
                     'Malaysia': 6, 'Philippines': 9, 'UK': 13, 'France': 2, 'Brazil': 0}
    categories_map = {'Semiconductors': 7, 'Passive Components': 5, 'Connectors': 0,
                      'PCB': 4, 'Displays': 1, 'Sensors': 6, 'Memory': 3, 'Power Management': 5}
    
    feature_vector = np.array([[
        req.years_in_business, req.num_manufacturing_sites, req.annual_revenue_millions,
        req.num_employees, req.on_time_delivery_rate, req.defect_rate_ppm, req.lead_time_days,
        req.lead_time_variability, req.financial_health_score, req.debt_to_equity_ratio,
        req.num_certifications, req.has_iso9001, req.has_iso14001, req.has_iatf16949,
        req.num_customers, req.single_source_pct, req.sub_tier_visibility,
        req.recent_disruption_events, req.news_sentiment_score, req.compliance_violations,
        req.geographic_risk_index,
        countries_map.get(req.country, 0),
        categories_map.get(req.component_category, 0),
        # Engineered features
        req.annual_revenue_millions * 1e6 / max(req.num_employees, 1),
        req.on_time_delivery_rate * (1 - req.lead_time_variability),
        req.has_iso9001 + req.has_iso14001 + req.has_iatf16949,
        req.single_source_pct * req.geographic_risk_index,
        req.financial_health_score / (1 + req.debt_to_equity_ratio),
        np.log1p(req.years_in_business) * req.num_manufacturing_sites,
        (1 - req.defect_rate_ppm / 10000) * req.on_time_delivery_rate,
        req.recent_disruption_events * (1 - req.news_sentiment_score),
    ]])
    
    model = models["supplier_risk"]["model"]
    pred = model.predict(feature_vector)[0]
    proba = model.predict_proba(feature_vector)[0]
    
    labels = ["High", "Low", "Medium"]
    pred_label = labels[pred]
    risk_score = round(float(proba[0] * 100 + proba[2] * 50), 1)
    
    factors = []
    if req.on_time_delivery_rate < 0.9: factors.append(f"Low delivery rate: {req.on_time_delivery_rate:.0%}")
    if req.financial_health_score < 50: factors.append(f"Weak finances: {req.financial_health_score}/100")
    if req.geographic_risk_index > 0.5: factors.append(f"High geo risk: {req.geographic_risk_index}")
    if req.defect_rate_ppm > 1000: factors.append(f"High defects: {req.defect_rate_ppm} PPM")
    if req.debt_to_equity_ratio > 2: factors.append(f"High debt: {req.debt_to_equity_ratio}")
    if req.compliance_violations > 0: factors.append(f"Violations: {req.compliance_violations}")
    if not factors: factors.append("No major risk factors detected")
    
    recs = {"High": "URGENT: Diversify supply base. Audit within 30 days.",
            "Medium": "Monitor quarterly. Review financials. Develop alternatives.",
            "Low": "Low risk. Standard annual review."}
    
    return SupplierRiskResponse(
        risk_label=pred_label, risk_score=risk_score,
        risk_probabilities={labels[i]: round(float(proba[i]), 4) for i in range(3)},
        top_risk_factors=factors[:5], recommendation=recs[pred_label]
    )


@app.post("/predict/component-obsolescence", response_model=ObsolescenceResponse)
async def predict_obsolescence(req: ObsolescenceRequest):
    if "obsolescence" not in models:
        raise HTTPException(503, "Model not loaded. Run project2 first.")
    
    stage_map = {'Active': 0, 'Mature': 1, 'Declining': 2, 'Last Buy': 3, 'EOL Announced': 4}
    lc = stage_map.get(req.lifecycle_stage, 1)
    
    feature_vector = np.array([[
        req.years_since_introduction, req.technology_node_nm, req.num_alternative_parts,
        req.num_authorized_distributors, req.monthly_demand_units, req.demand_trend_6m,
        req.demand_trend_12m, req.avg_lead_time_weeks, req.lead_time_increase_pct,
        req.price_trend_6m, req.num_pcn_notices, req.last_pcn_months_ago,
        req.manufacturer_financial_health, req.num_design_wins, req.cross_reference_count,
        req.rohs_compliant, req.automotive_qualified, req.military_grade,
        lc, 0, 0, 0,
        req.years_since_introduction * lc,
        req.num_authorized_distributors * req.num_alternative_parts,
        req.demand_trend_6m + req.demand_trend_12m,
        req.lead_time_increase_pct / (req.num_authorized_distributors + 1),
        lc * 10 + req.years_since_introduction + req.num_pcn_notices * 5 - req.demand_trend_12m * 20,
        1 if req.technology_node_nm >= 90 else 0,
        1 / (req.last_pcn_months_ago + 1),
        req.monthly_demand_units / (req.avg_lead_time_weeks + 1),
    ]])
    
    imputer = models["obsolescence"]["imputer"]
    scaler = models["obsolescence"]["scaler"]
    model = models["obsolescence"]["model"]
    
    feature_vector = imputer.transform(feature_vector)
    feature_vector = scaler.transform(feature_vector)
    eol_prob = float(model.predict_proba(feature_vector)[0][1])
    
    if eol_prob >= 0.7:
        level, actions = "CRITICAL", ["LAST-TIME BUY now", "Source alternatives", "Notify engineering", "Contact manufacturer"]
    elif eol_prob >= 0.4:
        level, actions = "HIGH", ["Evaluate alternatives", "Build safety stock", "Monitor PCN closely"]
    elif eol_prob >= 0.2:
        level, actions = "MEDIUM", ["Add to watchlist", "Identify replacements"]
    else:
        level, actions = "LOW", ["Standard monitoring"]
    
    return ObsolescenceResponse(
        eol_probability=round(eol_prob, 4),
        eol_prediction="EOL Likely" if eol_prob >= 0.5 else "Active (Safe)",
        risk_level=level,
        confidence="High" if eol_prob > 0.8 or eol_prob < 0.2 else "Medium",
        recommended_actions=actions
    )


@app.post("/predict/demand-forecast", response_model=DemandForecastResponse)
async def predict_demand(req: DemandForecastRequest):
    if "demand" not in models:
        raise HTTPException(503, "Model not loaded. Run project3 first.")
    
    d = req.recent_demands
    cats = {'Semiconductors': 4, 'Capacitors': 0, 'Resistors': 3, 'Connectors': 1, 'ICs': 2}
    
    ema = d[-1]
    for v in reversed(d[-3:]):
        ema = 0.5 * v + 0.5 * ema
    
    feature_vector = np.array([[
        req.month, (req.month - 1) // 3 + 1, cats.get(req.category, 0),
        d[-1], d[-2], d[-3], d[-6] if len(d) >= 6 else d[0],
        np.mean(d[-3:]), np.mean(d[-6:]),
        np.std(d[-3:]), np.std(d[-6:]),
        min(d[-3:]), max(d[-3:]),
        (d[-1] - d[-2]) / max(d[-2], 1), ema,
        req.avg_unit_price, req.avg_lead_time_weeks, req.num_orders, req.market_index,
        req.avg_unit_price, req.avg_lead_time_weeks,
    ]])
    
    pred = max(0, float(models["demand"]["model"].predict(feature_vector)[0]))
    margin = max(pred * np.std(d[-3:]) / max(np.mean(d[-3:]), 1), pred * 0.05)
    
    if d[-1] > d[-3] * 1.05: trend = f"Increasing (+{(d[-1]/d[-3]-1)*100:.1f}%)"
    elif d[-1] < d[-3] * 0.95: trend = f"Decreasing ({(d[-1]/d[-3]-1)*100:.1f}%)"
    else: trend = "Stable"
    
    return DemandForecastResponse(
        forecasted_demand=round(pred), model_used="Gradient Boosting (MAPE=4.3%)",
        confidence_interval={"lower": round(pred - margin), "upper": round(pred + margin)},
        trend=trend
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
