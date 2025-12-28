import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import yfinance as yf
from models import MultiSymbolTradingAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_training_data():
    """Télécharger données historiques pour entraînement initial"""
    
    # Données EURUSD (2018-2024)
    logger.info("Téléchargement données EURUSD...")
    eurusd = yf.download("EURUSD=X", start="2018-01-01", end="2024-01-01", interval="1h")
    
    # Données XAUUSD (Or)
    logger.info("Téléchargement données XAUUSD...")
    xauusd = yf.download("GC=F", start="2018-01-01", end="2024-01-01", interval="1h")
    
    return {
        "EURUSD": eurusd,
        "XAUUSD": xauusd
    }

def prepare_training_samples(data, symbol):
    """Préparer échantillons d'entraînement"""
    df = data[symbol]
    
    samples = []
    
    for i in range(100, len(df) - 50):
        # Features
        features = []
        
        # Prix
        current = df.iloc[i]
        features.extend([
            current['Close'],
            current['High'] - current['Low'],
            (current['Close'] - current['Open']) / current['Open'] if current['Open'] > 0 else 0
        ])
        
        # Momentum
        prev_10 = df.iloc[i-10:i]['Close'].values
        if len(prev_10) > 5:
            features.extend([
                (prev_10[0] - prev_10[4]) / prev_10[4],
                (prev_10[0] - prev_10[9]) / prev_10[9],
                np.std(prev_10)
            ])
        
        # Volume
        volumes = df.iloc[i-20:i]['Volume'].values
        features.append(current['Volume'] / np.mean(volumes) if np.mean(volumes) > 0 else 1)
        
        # RSI-like (simplifié)
        gains = []
        losses = []
        for j in range(1, 15):
            if i-j >= 0:
                change = df.iloc[i-j+1]['Close'] - df.iloc[i-j]['Close']
                if change > 0:
                    gains.append(change)
                else:
                    losses.append(abs(change))
        
        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 1
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi / 100)
        
        # Label (futur rendement sur 10 périodes)
        future_price = df.iloc[i+10]['Close']
        current_price = df.iloc[i]['Close']
        future_return = (future_price - current_price) / current_price
        
        # Classification
        if future_return > 0.002:  # > 0.2%
            label = "BUY"
        elif future_return < -0.002:  # < -0.2%
            label = "SELL"
        else:
            label = "HOLD"
        
        samples.append({
            'features': features[:20],  # Limiter à 20 features
            'label': label,
            'timestamp': df.index[i]
        })
    
    return samples

def train_initial_models():
    """Entraîner les modèles initiaux"""
    
    # Initialiser IA
    ai = MultiSymbolTradingAI()
    
    # Télécharger données
    data = download_training_data()
    
    for symbol in ["EURUSD", "XAUUSD"]:
        logger.info(f"Préparation données pour {symbol}...")
        samples = prepare_training_samples(data, symbol)
        
        if len(samples) < 100:
            logger.warning(f"Pas assez de données pour {symbol}")
            continue
        
        # Préparer X, y
        X = np.array([s['features'] for s in samples])
        y = np.array([0 if s['label'] == "BUY" else 1 if s['label'] == "SELL" else 2 for s in samples])
        
        # Entraînement
        logger.info(f"Entraînement modèle {symbol}...")
        ai.scalers[symbol].fit(X)
        X_scaled = ai.scalers[symbol].transform(X)
        ai.models[symbol].fit(X_scaled, y)
        
        # Sauvegarder
        model_path = f"models/{symbol}_model.pkl"
        scaler_path = f"models/{symbol}_scaler.pkl"
        
        joblib.dump(ai.models[symbol], model_path)
        joblib.dump(ai.scalers[symbol], scaler_path)
        
        logger.info(f"Modèle {symbol} sauvegardé: {len(samples)} échantillons")
    
    logger.info("✅ Entraînement initial terminé")

if __name__ == "__main__":
    train_initial_models()