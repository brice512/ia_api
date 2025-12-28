from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from datetime import datetime, timedelta
import uvicorn
import logging
from contextlib import asynccontextmanager
import json
import asyncio
from collections import defaultdict
import psutil
import os
from dotenv import load_dotenv

# Charger variables d'environnement
load_dotenv()

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Modèles Pydantic
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
    timestamp: str
    candles: List[Candle]
    features: List[float]
    indicators: Dict[str, float]
    market_context: Optional[Dict] = None
    signature: Optional[str] = None

class FeedbackRequest(BaseModel):
    bot_id: str
    symbol: str
    timeframe: str
    action: str
    price: float
    volume: float
    confidence: float
    timestamp: str
    features: List[float]
    market_condition: str
    result: str = "pending"

class HeartbeatRequest(BaseModel):
    bot_id: str
    timestamp: str
    exposure: float
    open_positions: int
    account_balance: float
    account_equity: float

# Classe modèle IA multi-symboles
class MultiSymbolTradingAI:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importances = {}
        self.training_data = defaultdict(list)
        self.performance_history = defaultdict(list)
        self.symbol_configs = self._load_symbol_configs()
        self.initialize_models()
        
    def _load_symbol_configs(self):
        """Configurations spécifiques par symbole"""
        return {
            "EURUSD": {
                "optimal_tf": "H1",
                "volatility_threshold": 0.0010,
                "confidence_threshold": 0.65,
                "features_count": 20,
                "model_type": "xgboost",
                "retrain_interval": 100
            },
            "XAUUSD": {
                "optimal_tf": "H2",
                "volatility_threshold": 0.0020,
                "confidence_threshold": 0.70,
                "features_count": 25,
                "model_type": "ensemble",
                "retrain_interval": 150
            }
        }
    
    def initialize_models(self):
        """Initialiser les modèles pour chaque symbole"""
        for symbol, config in self.symbol_configs.items():
            try:
                # Essayer de charger modèle sauvegardé
                model_path = f"models/{symbol}_model.pkl"
                scaler_path = f"models/{symbol}_scaler.pkl"
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.models[symbol] = joblib.load(model_path)
                    self.scalers[symbol] = joblib.load(scaler_path)
                    logger.info(f"Modèle chargé pour {symbol}")
                else:
                    # Créer nouveau modèle
                    if config["model_type"] == "ensemble":
                        self.models[symbol] = self._create_ensemble_model()
                    else:
                        self.models[symbol] = self._create_xgboost_model()
                    
                    self.scalers[symbol] = StandardScaler()
                    logger.info(f"Nouveau modèle créé pour {symbol}")
                    
                    # Créer dossier models si inexistant
                    os.makedirs("models", exist_ok=True)
                    
            except Exception as e:
                logger.error(f"Erreur initialisation modèle {symbol}: {e}")
                # Fallback au modèle par défaut
                self.models[symbol] = self._create_xgboost_model()
                self.scalers[symbol] = StandardScaler()
    
    def _create_xgboost_model(self):
        """Créer modèle XGBoost optimisé"""
        return xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.01,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            use_label_encoder=False
        )
    
    def _create_ensemble_model(self):
        """Créer modèle ensemble pour meilleure robustesse"""
        from sklearn.ensemble import VotingClassifier
        
        estimators = [
            ('xgb', self._create_xgboost_model()),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ]
        
        return VotingClassifier(estimators=estimators, voting='soft')
    
    def prepare_features(self, request: PredictionRequest) -> np.ndarray:
        """Préparer features spécifiques au symbole"""
        symbol = request.symbol
        candles = request.candles
        indicators = request.indicators
        
        # Features de base communes
        features = []
        
        # 1. Données de prix
        latest_candle = candles[0]
        features.extend([
            latest_candle.close,
            latest_candle.high - latest_candle.low,  # Range
            (latest_candle.close - latest_candle.open) / latest_candle.open if latest_candle.open != 0 else 0,  # Body %
        ])
        
        # 2. Momentum et tendance
        if len(candles) > 10:
            closes = [c.close for c in candles[:10]]
            features.extend([
                (closes[0] - closes[4]) / closes[4],  # 5-period return
                (closes[0] - closes[9]) / closes[9],  # 10-period return
                np.std(closes),  # Volatilité
            ])
        
        # 3. Volume analysis
        volumes = [c.volume for c in candles[:20]]
        if len(volumes) > 0:
            features.append(latest_candle.volume / np.mean(volumes) if np.mean(volumes) > 0 else 1)
        
        # 4. Technical indicators from MQL5
        if indicators:
            features.extend([
                indicators.get("rsi", 50) / 100,
                indicators.get("macd", 0),
                indicators.get("atr", 0) / latest_candle.close if latest_candle.close > 0 else 0,
            ])
        
        # 5. Features spécifiques EURUSD
        if "EURUSD" in symbol:
            # Caractéristiques forex
            features.extend([
                1 if 8 <= datetime.now().hour <= 17 else 0,  # Session Europe/US
                self._calculate_correlation_score(candles),  # Corrélation tendance
                self._detect_range_market(candles),  # Détection range
            ])
        
        # 6. Features spécifiques XAUUSD
        elif "XAUUSD" in symbol:
            # Caractéristiques or
            features.extend([
                1 if 14 <= datetime.now().hour <= 22 else 0,  # Meilleures heures
                self._calculate_gold_volatility(candles),  # Volatilité or
                self._detect_gold_gaps(candles),  # Détection gaps
                self._calculate_safe_haven_index(),  # Indice safe haven
            ])
        
        # 7. Contexte marché
        if request.market_context:
            features.extend([
                request.market_context.get("spread", 0),
                request.market_context.get("hour", 12) / 24,
                request.market_context.get("day_of_week", 1) / 7,
            ])
        
        # Pad features si nécessaire
        target_count = self.symbol_configs.get(symbol, {}).get("features_count", 20)
        while len(features) < target_count:
            features.append(0.0)
        
        return np.array(features[:target_count]).reshape(1, -1)
    
    def predict(self, request: PredictionRequest) -> tuple:
        """Faire une prédiction"""
        symbol = request.symbol
        config = self.symbol_configs.get(symbol, {})
        
        # Préparer features
        features = self.prepare_features(request)
        
        # Vérifier si modèle existe
        if symbol not in self.models:
            logger.warning(f"Modèle non trouvé pour {symbol}, utilisant modèle par défaut")
            symbol = "EURUSD"  # Fallback
        
        model = self.models[symbol]
        scaler = self.scalers[symbol]
        
        try:
            # Normalisation
            if not hasattr(scaler, 'mean_'):
                scaler.fit(features)
            features_scaled = scaler.transform(features)
            
            # Prédiction
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features_scaled)[0]
            else:
                # Pour VotingClassifier
                proba = model._predict_proba(features_scaled)[0]
            
            # Interprétation (3 classes: BUY, SELL, HOLD)
            if len(proba) >= 3:
                buy_prob = proba[0]
                sell_prob = proba[1]
                hold_prob = proba[2]
                
                # Seuil dynamique
                confidence_threshold = config.get("confidence_threshold", 0.65)
                
                if buy_prob > sell_prob and buy_prob > hold_prob and buy_prob > confidence_threshold:
                    prediction = "BUY"
                    confidence = float(buy_prob)
                elif sell_prob > buy_prob and sell_prob > hold_prob and sell_prob > confidence_threshold:
                    prediction = "SELL"
                    confidence = float(sell_prob)
                else:
                    prediction = "HOLD"
                    confidence = float(max(buy_prob, sell_prob, hold_prob))
            else:
                # Binary classification
                prediction = "BUY" if proba[0] > 0.5 else "SELL"
                confidence = float(max(proba[0], 1 - proba[0]))
            
            # Mettre à jour feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importances[symbol] = model.feature_importances_.tolist()
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Erreur prédiction {symbol}: {e}")
            return "HOLD", 0.5
    
    def learn_from_feedback(self, feedback: FeedbackRequest):
        """Apprendre du feedback"""
        symbol = feedback.symbol
        
        # Préparer données
        features = np.array(feedback.features).reshape(1, -1)
        
        # Stocker pour entraînement batch
        self.training_data[symbol].append({
            'features': features[0],
            'action': feedback.action,
            'confidence': feedback.confidence,
            'timestamp': feedback.timestamp,
            'market_condition': feedback.market_condition
        })
        
        # Vérifier si besoin de réentraînement
        config = self.symbol_configs.get(symbol, {})
        retrain_interval = config.get("retrain_interval", 100)
        
        if len(self.training_data[symbol]) >= retrain_interval:
            self._retrain_model(symbol)
    
    def _retrain_model(self, symbol: str):
        """Réentraîner le modèle pour un symbole"""
        try:
            data = self.training_data[symbol]
            if len(data) < 50:  # Minimum d'échantillons
                return
            
            df = pd.DataFrame(data)
            
            # Préparer features et labels
            X = np.array(df['features'].tolist())
            y = df['action'].apply(lambda x: 0 if x == "BUY" else 1 if x == "SELL" else 2).values
            
            # Normalisation
            self.scalers[symbol].fit(X)
            X_scaled = self.scalers[symbol].transform(X)
            
            # Entraînement
            self.models[symbol].fit(X_scaled, y)
            
            # Sauvegarder
            model_path = f"models/{symbol}_model.pkl"
            scaler_path = f"models/{symbol}_scaler.pkl"
            
            joblib.dump(self.models[symbol], model_path)
            joblib.dump(self.scalers[symbol], scaler_path)
            
            logger.info(f"Modèle réentraîné pour {symbol} avec {len(df)} échantillons")
            
            # Réinitialiser données d'entraînement
            self.training_data[symbol] = []
            
        except Exception as e:
            logger.error(f"Erreur réentraînement {symbol}: {e}")
    
    def _calculate_correlation_score(self, candles):
        """Calculer score de corrélation pour EURUSD"""
        if len(candles) < 20:
            return 0.5
        
        closes = [c.close for c in candles[:20]]
        returns = np.diff(closes) / closes[:-1]
        
        if len(returns) < 2:
            return 0.5
        
        # Autocorrélation lag 1
        correlation = np.corrcoef(returns[:-1], returns[1:])[0, 1]
        return abs(correlation) if not np.isnan(correlation) else 0.5
    
    def _detect_range_market(self, candles):
        """Détecter marché en range pour EURUSD"""
        if len(candles) < 20:
            return 0
        
        highs = [c.high for c in candles[:20]]
        lows = [c.low for c in candles[:20]]
        
        range_size = max(highs) - min(lows)
        avg_range = np.mean([h - l for h, l in zip(highs, lows)])
        
        return 1 if range_size < avg_range * 3 else 0
    
    def _calculate_gold_volatility(self, candles):
        """Calculer volatilité spécifique à l'or"""
        if len(candles) < 10:
            return 0.01
        
        returns = []
        for i in range(min(9, len(candles)-1)):
            ret = (candles[i].close - candles[i+1].close) / candles[i+1].close
            returns.append(abs(ret))
        
        return np.mean(returns) if returns else 0.01
    
    def _detect_gold_gaps(self, candles):
        """Détecter gaps fréquents dans l'or"""
        if len(candles) < 2:
            return 0
        
        gap = abs(candles[0].open - candles[1].close) / candles[1].close
        return 1 if gap > 0.002 else 0  # Gap > 0.2%
    
    def _calculate_safe_haven_index(self):
        """Indice safe haven simplifié (heure de trading)"""
        hour = datetime.now().hour
        # L'or est plus actif en période d'incertitude (après-midi US)
        return 1 if 14 <= hour <= 22 else 0

# Application FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie"""
    # Démarrage
    app.state.ai = MultiSymbolTradingAI()
    app.state.active_bots = {}
    app.state.start_time = datetime.now()
    
    logger.info("🚀 Trading AI API démarrée")
    logger.info(f"Modèles chargés: {list(app.state.ai.models.keys())}")
    
    # Tâche de fond pour nettoyage
    asyncio.create_task(cleanup_task(app))
    
    yield
    
    # Arrêt
    logger.info("Arrêt de l'API Trading AI")

async def cleanup_task(app: FastAPI):
    """Tâche de nettoyage périodique"""
    while True:
        await asyncio.sleep(3600)  # Toutes les heures
        # Nettoyer les bots inactifs (> 1 heure)
        current_time = datetime.now()
        inactive_bots = [
            bot_id for bot_id, last_seen in app.state.active_bots.items()
            if (current_time - last_seen).total_seconds() > 3600
        ]
        for bot_id in inactive_bots:
            del app.state.active_bots[bot_id]
            logger.info(f"Bot nettoyé: {bot_id}")

app = FastAPI(
    title="Trading AI Cloud API",
    description="API d'IA pour trading EURUSD/XAUUSD",
    version="3.0",
    lifespan=lifespan
)

# Routes API
@app.get("/")
async def root():
    """Page d'accueil"""
    return {
        "service": "Trading AI Cloud API",
        "version": "3.0",
        "status": "operational",
        "uptime": str(datetime.now() - app.state.start_time),
        "models_loaded": list(app.state.ai.models.keys())
    }

@app.get("/health")
async def health_check():
    """Vérifier santé de l'API"""
    memory = psutil.virtual_memory()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(app.state.ai.models),
        "active_bots": len(app.state.active_bots),
        "memory_usage": f"{memory.percent}%",
        "uptime": str(datetime.now() - app.state.start_time)
    }

@app.post("/api/v2/predict")
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Obtenir une prédiction de trading"""
    try:
        # Mettre à jour activité bot
        app.state.active_bots[request.bot_id] = datetime.now()
        
        # Faire la prédiction
        prediction, confidence = app.state.ai.predict(request)
        
        # Préparer réponse
        response = {
            "prediction": prediction,
            "confidence": confidence,
            "symbol": request.symbol,
            "timestamp": datetime.now().isoformat(),
            "model_version": "3.0",
            "feature_importance": app.state.ai.feature_importances.get(request.symbol, [])
        }
        
        # Tâche de fond pour analytics
        background_tasks.add_task(log_prediction, request, prediction, confidence)
        
        return response
        
    except Exception as e:
        logger.error(f"Erreur prédiction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/learn")
async def learn(feedback: FeedbackRequest, background_tasks: BackgroundTasks):
    """Apprendre du feedback"""
    try:
        # Apprentissage en arrière-plan
        background_tasks.add_task(app.state.ai.learn_from_feedback, feedback)
        
        return {
            "status": "learning_started",
            "symbol": feedback.symbol,
            "timestamp": datetime.now().isoformat(),
            "training_samples": len(app.state.ai.training_data.get(feedback.symbol, []))
        }
        
    except Exception as e:
        logger.error(f"Erreur apprentissage: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/heartbeat")
async def heartbeat(request: HeartbeatRequest):
    """Recevoir heartbeat des bots"""
    app.state.active_bots[request.bot_id] = datetime.now()
    
    return {
        "status": "acknowledged",
        "bot_id": request.bot_id,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v2/report")
async def receive_report(report: dict):
    """Recevoir rapport périodique"""
    logger.info(f"Rapport reçu: {report.get('bot_id', 'unknown')}")
    return {"status": "received"}

@app.post("/api/v2/final_report")
async def final_report(report: dict):
    """Recevoir rapport final"""
    logger.info(f"Rapport final reçu: {report.get('bot_id', 'unknown')}")
    return {"status": "received"}

@app.get("/api/v2/stats")
async def get_stats():
    """Obtenir statistiques de l'API"""
    return {
        "active_bots": len(app.state.active_bots),
        "models": {sym: type(model).__name__ for sym, model in app.state.ai.models.items()},
        "training_samples": {sym: len(data) for sym, data in app.state.ai.training_data.items()},
        "uptime": str(datetime.now() - app.state.start_time)
    }

@app.post("/api/v2/retrain/{symbol}")
async def retrain_model(symbol: str):
    """Forcer le réentraînement d'un modèle"""
    try:
        app.state.ai._retrain_model(symbol)
        return {
            "status": "retrained",
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def log_prediction(request: PredictionRequest, prediction: str, confidence: float):
    """Logger les prédictions pour analytics"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "bot_id": request.bot_id,
        "symbol": request.symbol,
        "timeframe": request.timeframe,
        "prediction": prediction,
        "confidence": confidence,
        "price": request.candles[0].close if request.candles else 0
    }
    
    # Sauvegarder dans fichier JSONL
    with open("predictions.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(
        app,
        host="0.0.zone.0",
        port=port,
        log_level="info"
    )