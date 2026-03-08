from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# ── Setup ──
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Z2Data ML Prediction API",
    description="Supply Chain ML Models - Supplier Risk, Obsolescence, Demand Forecast",
    version="1.0.0",
    docs_url="/docs"
)

# ── Load Models on Startup ──
models = {}

@app.on_event("startup")
async def load_models():
    """Load all ML models into memory at startup"""
    try:
        models["supplier_risk"] = {
            "model": joblib.load("models/project1/gradient_boosting_tuned.joblib"),
            "scaler": joblib.load("models/project1/scaler.joblib"),
            "imputer": joblib.load("models/project1/imputer.joblib"),
        }
        models["obsolescence"] = {
            "model": joblib.load("models/project2/best_model.joblib"),
            "scaler": joblib.load("models/project2/scaler.joblib"),
            "imputer": joblib.load("models/project2/imputer.joblib"),
        }
        models["demand_forecast"] = {
            "model": joblib.load("models/project3/best_forecasting_model.joblib"),
        }
        logger.info("All models loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading models: {e}")

# ══════════════════════════════════════════════════
# Request/Response Schemas (Pydantic)
# ══════════════════════════════════════════════════

class SupplierRiskRequest(BaseModel):
    """Input schema for supplier risk prediction"""
    years_in_business: int = Field(..., ge=1, le=100, description="Years supplier has been operating")
    num_manufacturing_sites: int = Field(..., ge=1, le=50)
    annual_revenue_millions: float = Field(..., ge=0)
    num_employees: int = Field(..., ge=1)
    on_time_delivery_rate: float = Field(..., ge=0, le=1)
    defect_rate_ppm: float = Field(..., ge=0)
    lead_time_days: int = Field(..., ge=1)
    lead_time_variability: float = Field(..., ge=0, le=1)
    financial_health_score: float = Field(..., ge=0, le=100)
    debt_to_equity_ratio: float = Field(..., ge=0)
    num_certifications: int = Field(..., ge=0)
    has_iso9001: int = Field(..., ge=0, le=1)
    has_iso14001: int = Field(..., ge=0, le=1)
    has_iatf16949: int = Field(..., ge=0, le=1)
    geographic_risk_index: float = Field(..., ge=0, le=1)
    country: str = Field(..., description="Supplier country")
    
    class Config:
        json_schema_extra = {
            "example": {
                "years_in_business": 15,
                "num_manufacturing_sites": 3,
                "annual_revenue_millions": 500,
                "num_employees": 2000,
                "on_time_delivery_rate": 0.92,
                "defect_rate_ppm": 300,
                "lead_time_days": 21,
                "lead_time_variability": 0.12,
                "financial_health_score": 72,
                "debt_to_equity_ratio": 0.8,
                "num_certifications": 4,
                "has_iso9001": 1,
                "has_iso14001": 1,
                "has_iatf16949": 0,
                "geographic_risk_index": 0.35,
                "country": "Taiwan"
            }
        }

class SupplierRiskResponse(BaseModel):
    risk_label: str
    risk_score: float
    risk_probabilities: dict
    top_risk_factors: List[str]
    recommendation: str

class ObsolescenceRequest(BaseModel):
    years_since_introduction: float
    technology_node_nm: int
    lifecycle_stage: str = Field(..., description="Active|Mature|Declining|Last Buy|EOL Announced")
    num_alternative_parts: int
    num_authorized_distributors: int
    demand_trend_12m: float
    lead_time_increase_pct: float
    num_pcn_notices: int
    manufacturer_financial_health: float

class ObsolescenceResponse(BaseModel):
    eol_probability: float
    eol_prediction: str
    risk_level: str
    recommended_actions: List[str]

class DemandForecastRequest(BaseModel):
    component_family: str
    recent_demands: List[float] = Field(..., min_length=6, description="Last 6+ months of demand")
    avg_unit_price: float
    avg_lead_time_weeks: float
    category: str

class DemandForecastResponse(BaseModel):
    forecasted_demand: float
    confidence_interval: dict
    trend: str

# ══════════════════════════════════════════════════
# API Endpoints
# ══════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers"""
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models/info")
async def model_info():
    """Get information about deployed models"""
    return {
        "models": {
            "supplier_risk": {
                "description": "Classifies suppliers into Low/Medium/High risk",
                "algorithm": "Gradient Boosting (tuned)",
                "features": 31,
                "accuracy": "84.5%",
                "endpoint": "/predict/supplier-risk"
            },
            "obsolescence": {
                "description": "Predicts if component will go EOL in 12 months",
                "algorithm": "Logistic Regression (balanced)",
                "features": 30,
                "auc_roc": "0.991",
                "endpoint": "/predict/component-obsolescence"
            },
            "demand_forecast": {
                "description": "Forecasts next month component demand",
                "algorithm": "Gradient Boosting Regressor",
                "features": 21,
                "mape": "4.3%",
                "endpoint": "/predict/demand-forecast"
            }
        }
    }

@app.post("/predict/supplier-risk", response_model=SupplierRiskResponse)
async def predict_supplier_risk(request: SupplierRiskRequest):
    """Predict supplier risk level and score"""
    try:
        # Feature engineering (same as training pipeline)
        features = {
            **request.dict(),
            "revenue_per_employee": request.annual_revenue_millions * 1e6 / request.num_employees,
            "delivery_reliability": request.on_time_delivery_rate * (1 - request.lead_time_variability),
            "certification_score": request.has_iso9001 + request.has_iso14001 + request.has_iatf16949,
            "risk_exposure": request.geographic_risk_index * 0.3,  # simplified
            "financial_stability": request.financial_health_score / (1 + request.debt_to_equity_ratio),
        }
        
        # Predict
        model = models["supplier_risk"]["model"]
        # ... (feature vector construction and prediction)
        
        labels = ["High", "Low", "Medium"]
        probas = [0.15, 0.45, 0.40]  # placeholder
        pred_label = labels[1]
        
        return SupplierRiskResponse(
            risk_label=pred_label,
            risk_score=round((1 - probas[1]) * 100, 1),
            risk_probabilities=dict(zip(labels, [round(p, 3) for p in probas])),
            top_risk_factors=["geographic_risk_index", "financial_health_score"],
            recommendation="Monitor quarterly. Consider diversifying supply base."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/component-obsolescence", response_model=ObsolescenceResponse)
async def predict_obsolescence(request: ObsolescenceRequest):
    """Predict component end-of-life probability"""
    try:
        model = models["obsolescence"]["model"]
        # ... (feature construction and prediction)
        
        eol_prob = 0.15  # placeholder
        
        actions = []
        if eol_prob > 0.7:
            actions = ["Initiate last-time buy", "Source alternatives immediately", "Notify engineering team"]
        elif eol_prob > 0.3:
            actions = ["Evaluate alternative components", "Monitor manufacturer announcements"]
        else:
            actions = ["Continue standard monitoring"]
            
        return ObsolescenceResponse(
            eol_probability=round(eol_prob, 3),
            eol_prediction="EOL Likely" if eol_prob > 0.5 else "Active",
            risk_level="High" if eol_prob > 0.7 else "Medium" if eol_prob > 0.3 else "Low",
            recommended_actions=actions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/demand-forecast", response_model=DemandForecastResponse)
async def predict_demand(request: DemandForecastRequest):
    """Forecast next month demand for a component"""
    try:
        model = models["demand_forecast"]["model"]
        demands = request.recent_demands
        
        # Calculate time series features
        forecast = np.mean(demands[-3:]) * 1.02  # placeholder
        
        trend = "increasing" if demands[-1] > demands[-3] else "decreasing" if demands[-1] < demands[-3] else "stable"
        
        return DemandForecastResponse(
            forecasted_demand=round(forecast, 0),
            confidence_interval={
                "lower": round(forecast * 0.9, 0),
                "upper": round(forecast * 1.1, 0)
            },
            trend=trend
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))