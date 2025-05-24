"""
Module for detecting anomalies in transaction data using Isolation Forest.
"""
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def initialize_isolation_forest():
    """
    Initialize the Isolation Forest model.
    
    Returns:
        IsolationForest: Initialized model
    """
    return IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=0.014,  # This will generate approximately 14 anomalies out of 1000 transactions
        random_state=42
    )

def prepare_features(data):
    """
    Prepare features for anomaly detection.
    
    Args:
        data (pandas.DataFrame): Raw transaction data
        
    Returns:
        pandas.DataFrame: Prepared features for anomaly detection
    """
    # Create a copy of the data
    features = data.copy()
    
    # Convert date to numeric features
    features['hora'] = features['fecha'].dt.hour
    features['dia_semana'] = features['fecha'].dt.dayofweek
    features['fin_semana'] = features['fecha'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # One-hot encode categorical features
    tipo_transaccion_dummies = pd.get_dummies(features['tipo_transaccion'], prefix='tipo')
    ubicacion_dummies = pd.get_dummies(features['ubicacion'], prefix='ubicacion')
    
    # Select numeric features
    numeric_features = [
        'monto',
        'duracion_segundos',
        'transacciones_recientes_24h',
        'saldo_anterior',
        'comision',
        'distancia_km',
        'intentos_fallidos',
        'hora',
        'dia_semana',
        'fin_semana'
    ]
    
    # Combine numeric features with encoded categorical features
    features = pd.concat([
        features[numeric_features],
        tipo_transaccion_dummies,
        ubicacion_dummies
    ], axis=1)
    
    # Calculate rolling statistics for time-based features
    features['monto_rolling_mean'] = features['monto'].rolling(window=10, min_periods=1).mean()
    features['monto_rolling_std'] = features['monto'].rolling(window=10, min_periods=1).std()
    features['tiempo_entre_transacciones'] = features['hora'].diff().abs()
    
    # Fill NaN values
    features = features.fillna(0)
    
    return features

def train_model(model, data):
    """
    Train the anomaly detection model.
    
    Args:
        model (IsolationForest): The model to train
        data (pandas.DataFrame): Training data
    
    Returns:
        IsolationForest: Trained model
    """
    # Prepare features
    features = prepare_features(data)
    
    # Train the model
    model.fit(features)
    
    return model

def predict_anomalies(model, data):
    """
    Predict anomalies in the data.
    
    Args:
        model (IsolationForest): Trained model
        data (pandas.DataFrame): Data to predict on
    
    Returns:
        pandas.DataFrame: Original data with anomaly predictions
    """
    # Prepare features
    features = prepare_features(data)
    
    # Get predictions
    predictions = model.predict(features)
    scores = model.score_samples(features)
    
    # Convert predictions to boolean (True for anomalies)
    is_anomaly = predictions == -1
    
    # Add predictions to original data
    result = data.copy()
    result['es_anomalia'] = is_anomaly
    result['puntaje_anomalia'] = -scores  # Convert to positive scores where higher means more anomalous
    
    return result 