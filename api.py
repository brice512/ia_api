from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from datetime import datetime
import uvicorn
import logging
import os
import asyncio
from contextlib import asynccontextmanager
from collections import defaultdict
import json

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TradingAI")

# ===================== MODELS =====================

class Candle(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float
    time: float

class PredictionRequest(BaseModel):
    bot_id: str
    symbol: str
    timeframe: str
    candles: List[Candle]
    indicators: Dict[str, float]
    features: List[float]

class FeedbackRequest(BaseModel):
    bot_id: str
    symbol: str
    action: str
    confidence: float
    features: List[float]
    result: str

# ===================== AI CORE =====================

class TradingAI:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.memory = defaultdict(list)
        self._init_models()

    def _init_models(self):
        for symbol in ["EURUSD", "XAUUSD"]:
            self.models[symbol] = GradientBoostingClassifier(n_estimators=150)
            self.scalers[symbol] = StandardScaler()
            logger.info(f"Model initialized for {symbol}")

    def predict(self, req: PredictionRequest):
        X = np.array(req.features).reshape(1, -1)
        scaler = self.scalers[req.symbol]
        model = self.models[req.symbol]

        if not hasattr(scaler, "mean_"):
            scaler.fit(X)

        Xs = scaler.transform(X)

        if not hasattr(model, "estimators_"):
            model.fit(Xs, [0])

        proba = model.predict_proba(Xs)[0]

        if proba[0] > 0.6:
            return "BUY", float(proba[0])
        elif proba[1] > 0.6:
            return "SELL", float(proba[1])
        return "HOLD", float(max(proba))

    def learn(self, fb: FeedbackRequest):
        y = 0 if fb.result == "win" else 1
        self.memory[fb.symbol].append((fb.features, y))

        if len(self.memory[fb.symbol]) >= 50:
            X, Y = zip(*self.memory[fb.symbol])
            X = np.array(X)
            scaler = self.scalers[fb.symbol]
            Xs = scaler.fit_transform(X)
            self.models[fb.symbol].fit(Xs, Y)
            self.memory[fb.symbol] = []
            logger.info(f"Model retrained for {fb.symbol}")

# ===================== APP =====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ai = TradingAI()
    app.state.start_time = datetime.now()
    yield

app = FastAPI(title="Trading AI API", lifespan=lifespan)

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/predict")
def predict(req: PredictionRequest):
    try:
        pred, conf = app.state.ai.predict(req)
        return {"prediction": pred, "confidence": conf}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/learn")
def learn(req: FeedbackRequest):
    app.state.ai.learn(req)
    return {"status": "learning"}

# ===================== RUN =====================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
