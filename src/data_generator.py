"""
Module for generating synthetic transaction data using Faker.
"""
from faker import Faker
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Initialize Faker with Chilean Spanish locale
fake = Faker('es_CL')

def generate_single_transaction(base_date=None, last_transaction_time=None):
    """
    Generate a single transaction record with realistic patterns.
    
    Args:
        base_date (datetime, optional): Base date for transaction timing
        last_transaction_time (datetime, optional): Time of last transaction
        
    Returns:
        dict: A dictionary containing transaction details
    """
    # Generate transaction time
    if base_date is None:
        base_date = fake.date_time_this_year()
    
    if last_transaction_time is None:
        transaction_time = base_date
    else:
        # Add some randomness to time between transactions
        # Most transactions happen within 24 hours of the last one
        hours_since_last = np.random.exponential(12)  # Exponential distribution for time gaps
        transaction_time = last_transaction_time + timedelta(hours=hours_since_last)
    
    # Generate amount with realistic patterns
    base_amount = np.random.lognormal(mean=10, sigma=1)  # Base amount in CLP
    
    # Adjust amount based on time of day
    hour = transaction_time.hour
    if 9 <= hour <= 17:  # Business hours
        amount_multiplier = 1.2
    elif 18 <= hour <= 22:  # Evening
        amount_multiplier = 1.5
    else:  # Night/early morning
        amount_multiplier = 0.8
    
    # Occasionally generate very large amounts (1% chance)
    if random.random() < 0.01:
        amount_multiplier *= 10
    
    amount = base_amount * amount_multiplier
    
    # Generate transaction type with realistic patterns
    transaction_types = ['transfer', 'withdrawal', 'deposit', 'payment']
    weights = [0.4, 0.3, 0.2, 0.1]  # More transfers and withdrawals than deposits
    transaction_type = random.choices(transaction_types, weights=weights)[0]
    
    # Generate location with realistic patterns
    locations = [
        'Santiago', 'Valparaíso', 'Concepción', 'La Serena', 'Antofagasta',
        'Temuco', 'Rancagua', 'Talca', 'Arica', 'Puerto Montt'
    ]
    # Higher probability for Santiago
    location_weights = [0.4] + [0.067] * 9
    location = random.choices(locations, weights=location_weights)[0]
    
    # Generate additional numeric features
    
    # 1. Transaction duration (in seconds)
    # Most transactions take between 30 seconds and 5 minutes
    duration = np.random.gamma(shape=2, scale=60)  # Mean around 2 minutes
    
    # 2. Number of previous transactions in the last 24 hours
    # Most accounts have 0-3 transactions per day
    recent_transactions = np.random.poisson(lam=2)
    
    # 3. Account balance before transaction
    # Balance typically between 100,000 and 10,000,000 CLP
    balance = np.random.lognormal(mean=12, sigma=1)
    
    # 4. Transaction fee
    # Fee is typically 0.1% to 1% of amount
    fee_percentage = np.random.uniform(0.001, 0.01)
    fee = amount * fee_percentage
    
    # 5. Distance from last transaction (in km)
    # Most transactions are within 10km of last location
    distance = np.random.exponential(scale=5)
    
    # 6. Number of failed attempts before success
    # Most transactions succeed on first try
    failed_attempts = np.random.geometric(p=0.9) - 1
    
    # 7. Device trust score (0-100)
    # Most devices have high trust scores
    device_trust = np.random.beta(a=5, b=1) * 100
    
    # 8. IP risk score (0-100)
    # Most IPs have low risk scores
    ip_risk = np.random.beta(a=1, b=5) * 100
    
    return {
        'transaction_id': fake.uuid4(),
        'date': transaction_time,
        'amount': round(amount, 2),
        'account_number': fake.bban(),
        'transaction_type': transaction_type,
        'location': location,
        'duration_seconds': round(duration, 2),
        'recent_transactions_24h': recent_transactions,
        'balance_before': round(balance, 2),
        'transaction_fee': round(fee, 2),
        'distance_km': round(distance, 2),
        'failed_attempts': failed_attempts,
        'device_trust_score': round(device_trust, 2),
        'ip_risk_score': round(ip_risk, 2)
    }

def generate_fake_transactions(num_transactions=1000, start_date=None):
    """
    Generate fake transaction data with realistic patterns.
    
    Args:
        num_transactions (int): Number of transactions to generate
        start_date (datetime): Start date for transactions
    
    Returns:
        pandas.DataFrame: Generated transaction data
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    
    # Generate timestamps with realistic patterns
    timestamps = []
    current_time = start_date
    
    # Generate more transactions during business hours
    for _ in range(num_transactions):
        # Add random time between 1 minute and 4 hours
        time_delta = timedelta(
            minutes=random.randint(1, 240)
        )
        current_time += time_delta
        timestamps.append(current_time)
    
    # Generate transaction amounts with realistic patterns
    amounts = []
    for timestamp in timestamps:
        hour = timestamp.hour
        # Higher amounts during business hours
        if 9 <= hour <= 17:
            base_amount = np.random.lognormal(mean=10, sigma=1)
        else:
            base_amount = np.random.lognormal(mean=9, sigma=1)
        # Round to integer
        amounts.append(int(round(base_amount)))
    
    # Generate transaction types with realistic distribution
    tipos_transaccion = ['compra', 'retiro', 'transferencia', 'deposito', 'pago']
    probabilidades = [0.4, 0.2, 0.2, 0.1, 0.1]  # More purchases, fewer deposits
    tipos = np.random.choice(tipos_transaccion, size=num_transactions, p=probabilidades)
    
    # Generate locations with realistic patterns
    ubicaciones = ['Santiago', 'Valparaíso', 'Concepción', 'La Serena', 'Antofagasta']
    probabilidades_ubicacion = [0.4, 0.2, 0.2, 0.1, 0.1]  # More transactions in Santiago
    ubicaciones_transacciones = np.random.choice(ubicaciones, size=num_transactions, p=probabilidades_ubicacion)
    
    # Generate additional numeric features
    duracion_segundos = np.random.gamma(shape=2, scale=30, size=num_transactions)  # Most transactions take 30-60 seconds
    transacciones_recientes_24h = np.random.poisson(lam=3, size=num_transactions)  # Average 3 transactions per 24h
    saldo_anterior = np.random.lognormal(mean=11, sigma=1, size=num_transactions)  # Log-normal distribution for balances
    comision = np.array(amounts) * np.random.uniform(0.01, 0.03, size=num_transactions)  # 1-3% commission
    distancia_km = np.random.exponential(scale=10, size=num_transactions)  # Exponential distribution for distances
    intentos_fallidos = np.random.geometric(p=0.8, size=num_transactions) - 1  # Most transactions succeed on first try
    
    # Round monetary values to integers
    saldo_anterior = np.round(saldo_anterior).astype(int)
    comision = np.round(comision).astype(int)
    
    # Create DataFrame with specified features
    data = pd.DataFrame({
        'fecha': timestamps,
        'monto': amounts,
        'tipo_transaccion': tipos,
        'ubicacion': ubicaciones_transacciones,
        'duracion_segundos': duracion_segundos,
        'transacciones_recientes_24h': transacciones_recientes_24h,
        'saldo_anterior': saldo_anterior,
        'comision': comision,
        'distancia_km': distancia_km,
        'intentos_fallidos': intentos_fallidos
    })
    
    return data

def save_fake_data(data, filename='data/generated_data.csv'):
    """
    Save generated data to CSV file.
    
    Args:
        data (pandas.DataFrame): Data to save
        filename (str): Output filename
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Convert datetime to string format
    data_to_save = data.copy()
    data_to_save['fecha'] = data_to_save['fecha'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Save to CSV with proper encoding and quoting
    data_to_save.to_csv(filename, index=False, encoding='utf-8', quoting=1)
    print(f"Datos de muestra generados exitosamente y guardados en {filename}")

if __name__ == "__main__":
    # Generate and save sample data
    data = generate_fake_transactions()
    save_fake_data(data) 