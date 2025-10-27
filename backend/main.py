"""
FastAPI Backend for Insider Trading ML Model
Provides REST API endpoints for model predictions and analytics
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Insider Trading ML API",
    description="API for predicting insider trading risk using machine learning (v3.0 Enhanced)",
    version="3.0.0"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and scaler
MODEL = None
SCALER = None
FEATURE_COLS = None
METRICS = None

# ================================================================
# Data Models (Pydantic)
# ================================================================

class TradeInput(BaseModel):
    """Input schema for a single trade prediction"""
    ticker_symbol: str = Field(..., example="AAPL")
    company_name: str = Field(..., example="Apple Inc.")
    insider_role: str = Field(..., example="Chief Executive Officer")
    under_schedule: bool = Field(..., example=False, description="Under 10b5-1 trading plan")
    aggregated_signal: str = Field(..., example="sell")
    aggregated_shares: float = Field(..., example=100000)
    aggregated_value_usd: float = Field(..., example=15000000)
    aggregated_percent_of_shares: float = Field(..., example=0.05)
    

class BulkTradeInput(BaseModel):
    """Input schema for bulk predictions"""
    trades: List[TradeInput]


class PredictionResponse(BaseModel):
    """Response schema for prediction"""
    ticker_symbol: str
    company_name: str
    insider_role: str
    predicted_high_risk: int
    risk_probability: float
    risk_level: str  # "Low", "Medium", "High", "Critical"


class ModelMetrics(BaseModel):
    """Response schema for model metrics"""
    model_version: str
    model_type: str
    train_date: str
    training_accuracy: float
    test_accuracy: float
    roc_auc: float
    features_used: List[str]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: str


# ================================================================
# Helper Functions
# ================================================================

def load_model():
    """Load the trained model and scaler"""
    global MODEL, SCALER, FEATURE_COLS, METRICS
    
    try:
        # Get parent directory (project root)
        base_dir = Path(__file__).parent.parent
        
        # Load model - prioritize improved model
        model_path = base_dir / "outputs" / "insider_trading_model_improved.pkl"
        if not model_path.exists():
            model_path = base_dir / "insider_trading_model.pkl"
        if not model_path.exists():
            model_path = base_dir / "insider_trading_model_v2.pkl"
        
        if model_path.exists():
            MODEL = joblib.load(model_path)
            print(f"✓ Model loaded from: {model_path}")
        else:
            print("⚠️  Model file not found!")
            return False
        
        # Load scaler (optional)
        scaler_path = base_dir / "feature_scaler.pkl"
        if not scaler_path.exists():
            scaler_path = base_dir / "feature_scaler_v2.pkl"
        
        if scaler_path.exists():
            SCALER = joblib.load(scaler_path)
            print(f"✓ Scaler loaded from: {scaler_path}")
        
        # Load metrics - prioritize improved metrics
        metrics_path = base_dir / "outputs" / "metrics_improved.json"
        if not metrics_path.exists():
            metrics_path = base_dir / "metrics.json"
        if not metrics_path.exists():
            metrics_path = base_dir / "metrics_v2.json"
        
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                METRICS = json.load(f)
            FEATURE_COLS = METRICS.get("features_used", [])
            print(f"✓ Metrics loaded from: {metrics_path}")
        else:
            # Default feature columns if metrics not found (v3.0 with new features)
            FEATURE_COLS = [
                "role_weight", "no_plan", "abs_value_usd", "abs_percent_shares",
                "abs_shares", "log_value_usd", "log_shares", "p_value_tkr",
                "p_percent_tkr", "p_shares_tkr", "is_buy", "is_sell",
                "value_x_role", "percent_x_role", "log_value_x_role", "high_value_no_plan",
                "is_director_level", "size_deviation_pct", "company_z_score",
                "quarter_end", "trades_same_day", "director_large_trade"
            ]
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False


def get_role_weight(role: str) -> float:
    """Calculate role weight based on insider role"""
    ROLE_WEIGHTS = {
        "chief executive officer": 1.0, "ceo": 1.0,
        "chief financial officer": 0.95, "cfo": 0.95,
        "chief operating officer": 0.9, "coo": 0.9,
        "chair": 0.9, "chairman": 0.9,
        "principal accounting officer": 0.85,
        "chief accounting officer": 0.85,
        "general counsel": 0.85,
        "chief legal officer": 0.85,
        "president": 0.85,
        "vice president": 0.75, "vp": 0.75,
        "director": 0.70
    }
    
    if not isinstance(role, str):
        return 0.6
    
    role_lower = role.lower()
    for key, weight in ROLE_WEIGHTS.items():
        if key in role_lower:
            return weight
    return 0.6


def create_features(trade: TradeInput) -> Dict[str, float]:
    """Create features from trade input (v3.0 with enhanced features)"""
    
    # Role weight
    role_weight = get_role_weight(trade.insider_role)
    
    # Transaction characteristics
    abs_value_usd = abs(trade.aggregated_value_usd)
    abs_percent_shares = abs(trade.aggregated_percent_of_shares)
    abs_shares = abs(trade.aggregated_shares)
    
    # Log transforms
    log_value_usd = np.log1p(abs_value_usd)
    log_shares = np.log1p(abs_shares)
    
    # Trading plan
    no_plan = 0 if trade.under_schedule else 1
    
    # Signal
    signal_lower = trade.aggregated_signal.lower()
    is_buy = 1 if signal_lower == "buy" else 0
    is_sell = 1 if signal_lower == "sell" else 0
    
    # Percentile ranks (simplified - using placeholder values)
    # In production, these would be calculated against historical data
    p_value_tkr = 0.5  # Placeholder
    p_percent_tkr = 0.5  # Placeholder
    p_shares_tkr = 0.5  # Placeholder
    
    # Interaction features
    value_x_role = abs_value_usd * role_weight
    percent_x_role = abs_percent_shares * role_weight
    log_value_x_role = log_value_usd * role_weight
    high_value_no_plan = (1 if abs_value_usd > 1000000 else 0) * no_plan
    
    # NEW v3.0 features
    # Director-level indicator
    is_director_level = 1 if any(kw in trade.insider_role.lower() for kw in 
                                  ["director", "chair", "ceo", "cfo", "coo"]) else 0
    
    # Size deviation (simplified - comparing to average trade size)
    avg_trade_size = 1000000  # Placeholder for average trade size
    size_deviation_pct = (abs_value_usd - avg_trade_size) / (avg_trade_size + 1e-6)
    size_deviation_pct = max(min(size_deviation_pct, 10), -10)  # Clip extreme values
    
    # Company z-score (placeholder - would need historical company data)
    company_z_score = 0.0  # Placeholder
    
    # Quarter end indicator (placeholder - would need actual execution date)
    quarter_end = 0  # Placeholder
    
    # Trades same day (placeholder - would need to query concurrent trades)
    trades_same_day = 1  # Placeholder
    
    # Director large trade
    director_large_trade = 1 if (is_director_level == 1 and abs_value_usd > 1000000) else 0
    
    # Return features in correct order
    return {
        "role_weight": role_weight,
        "no_plan": no_plan,
        "abs_value_usd": abs_value_usd,
        "abs_percent_shares": abs_percent_shares,
        "abs_shares": abs_shares,
        "log_value_usd": log_value_usd,
        "log_shares": log_shares,
        "p_value_tkr": p_value_tkr,
        "p_percent_tkr": p_percent_tkr,
        "p_shares_tkr": p_shares_tkr,
        "is_buy": is_buy,
        "is_sell": is_sell,
        "value_x_role": value_x_role,
        "percent_x_role": percent_x_role,
        "log_value_x_role": log_value_x_role,
        "high_value_no_plan": high_value_no_plan,
        "is_director_level": is_director_level,
        "size_deviation_pct": size_deviation_pct,
        "company_z_score": company_z_score,
        "quarter_end": quarter_end,
        "trades_same_day": trades_same_day,
        "director_large_trade": director_large_trade
    }


def get_risk_level(probability: float) -> str:
    """Convert probability to risk level"""
    if probability >= 0.8:
        return "Critical"
    elif probability >= 0.6:
        return "High"
    elif probability >= 0.4:
        return "Medium"
    else:
        return "Low"


# ================================================================
# API Endpoints
# ================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    print("\n" + "=" * 70)
    print("INSIDER TRADING ML API - STARTING")
    print("=" * 70)
    success = load_model()
    if success:
        print("✓ API ready to serve predictions")
    else:
        print("⚠️  API started but model not loaded. Check file paths.")
    print("=" * 70 + "\n")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Insider Trading ML API v3.0 - Enhanced Model",
        "documentation": "/docs",
        "health": "/health",
        "model_version": "3.0",
        "enhancements": "Optimal threshold tuning, 6 new features, precision-focused training"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if MODEL is not None else "degraded",
        "model_loaded": MODEL is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/metrics", response_model=ModelMetrics)
async def get_metrics():
    """Get model performance metrics"""
    if METRICS is None:
        raise HTTPException(status_code=503, detail="Metrics not available")
    
    return {
        "model_version": METRICS.get("model_version", "unknown"),
        "model_type": METRICS.get("model_type", "unknown"),
        "train_date": METRICS.get("train_date", "unknown"),
        "training_accuracy": METRICS.get("training_accuracy", 0.0),
        "test_accuracy": METRICS.get("test_accuracy", 0.0),
        "roc_auc": METRICS.get("roc_auc", 0.0),
        "features_used": METRICS.get("features_used", [])
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(trade: TradeInput):
    """Predict risk for a single trade"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create features
        features = create_features(trade)
        
        # Convert to DataFrame with correct column order
        X = pd.DataFrame([features])[FEATURE_COLS]
        
        # Scale if scaler is available
        if SCALER is not None:
            X = SCALER.transform(X)
        
        # Predict
        prediction = MODEL.predict(X)[0]
        probability = MODEL.predict_proba(X)[0, 1]
        
        return {
            "ticker_symbol": trade.ticker_symbol,
            "company_name": trade.company_name,
            "insider_role": trade.insider_role,
            "predicted_high_risk": int(prediction),
            "risk_probability": float(probability),
            "risk_level": get_risk_level(float(probability))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/bulk", response_model=List[PredictionResponse])
async def predict_bulk(data: BulkTradeInput):
    """Predict risk for multiple trades"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        
        for trade in data.trades:
            # Create features
            features = create_features(trade)
            
            # Convert to DataFrame
            X = pd.DataFrame([features])[FEATURE_COLS]
            
            # Scale if available
            if SCALER is not None:
                X = SCALER.transform(X)
            
            # Predict
            prediction = MODEL.predict(X)[0]
            probability = MODEL.predict_proba(X)[0, 1]
            
            results.append({
                "ticker_symbol": trade.ticker_symbol,
                "company_name": trade.company_name,
                "insider_role": trade.insider_role,
                "predicted_high_risk": int(prediction),
                "risk_probability": float(probability),
                "risk_level": get_risk_level(float(probability))
            })
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk prediction error: {str(e)}")


@app.get("/predictions/top/{n}")
async def get_top_predictions(n: int = 100):
    """Get top N high-risk predictions from saved file"""
    try:
        base_dir = Path(__file__).parent.parent
        
        # Try to load top predictions file - prioritize improved predictions
        file_path = base_dir / "outputs" / "insider_predictions_top100_improved.csv"
        if not file_path.exists():
            file_path = base_dir / "insider_predictions_top100.csv"
        if not file_path.exists():
            file_path = base_dir / "insider_predictions_top100_v2.csv"
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Predictions file not found")
        
        df = pd.read_csv(file_path)
        
        # Get top N
        df_top = df.head(n)
        
        # Convert to list of dicts
        results = df_top.to_dict('records')
        
        return {
            "count": len(results),
            "predictions": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading predictions: {str(e)}")


@app.get("/statistics")
async def get_statistics():
    """Get dataset statistics"""
    try:
        base_dir = Path(__file__).parent.parent
        
        # Load full predictions - prioritize improved predictions
        file_path = base_dir / "outputs" / "insider_predictions_improved.csv"
        if not file_path.exists():
            file_path = base_dir / "insider_predictions_full.csv"
        if not file_path.exists():
            file_path = base_dir / "insider_predictions_full_v2.csv"
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Predictions file not found")
        
        df = pd.read_csv(file_path)
        
        # Calculate statistics
        total_records = len(df)
        
        # Risk distribution
        if 'risk_probability' in df.columns:
            risk_dist = {
                "critical": int((df['risk_probability'] >= 0.8).sum()),
                "high": int(((df['risk_probability'] >= 0.6) & (df['risk_probability'] < 0.8)).sum()),
                "medium": int(((df['risk_probability'] >= 0.4) & (df['risk_probability'] < 0.6)).sum()),
                "low": int((df['risk_probability'] < 0.4).sum())
            }
            avg_risk = float(df['risk_probability'].mean())
        else:
            risk_dist = {"critical": 0, "high": 0, "medium": 0, "low": total_records}
            avg_risk = 0.0
        
        # Top companies
        if 'ticker_symbol' in df.columns:
            top_companies = df['ticker_symbol'].value_counts().head(10).to_dict()
        else:
            top_companies = {}
        
        return {
            "total_records": total_records,
            "risk_distribution": risk_dist,
            "average_risk_probability": avg_risk,
            "top_companies_by_volume": top_companies
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating statistics: {str(e)}")


@app.get("/feature-importance")
async def get_feature_importance():
    """Get feature importance rankings"""
    try:
        base_dir = Path(__file__).parent.parent
        
        file_path = base_dir / "feature_importance.csv"
        if not file_path.exists():
            file_path = base_dir / "feature_importance_v2.csv"
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Feature importance file not found")
        
        df = pd.read_csv(file_path)
        
        # Convert to list of dicts
        features = df.to_dict('records')
        
        return {
            "count": len(features),
            "features": features
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading feature importance: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
