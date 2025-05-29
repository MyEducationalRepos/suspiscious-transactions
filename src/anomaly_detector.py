"""
Module for detecting anomalies in transaction data using Isolation Forest.
"""
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def initialize_isolation_forest():
    """
    Initialize the Isolation Forest model.
    
    Returns:
        IsolationForest: Initialized model
    """
    return IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=0.01,  # 1% of transactions will be marked as anomalies
        random_state=42
    )

def prepare_features(data):
    """
    Prepare features for anomaly detection.
    
    Args:
        data (pandas.DataFrame): Raw transaction data
        
    Returns:
        numpy.ndarray: Prepared features for the model
    """
    # Select only the features we need
    features = data[['amount', 'failed_attempts']].copy()
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features

def train_model(model, data):
    """
    Train the Isolation Forest model on the data.
    
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
    result['is_anomaly'] = is_anomaly
    result['anomaly_score'] = -scores  # Convert to positive scores where higher means more anomalous
    
    return result 