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

def generate_single_transaction(transaction_id):
    """
    Generate a single transaction record with ID, amount and failed attempts.
    
    Args:
        transaction_id (int): Auto-incremented transaction ID
        
    Returns:
        dict: A dictionary containing transaction details
    """
    # Generate amount with realistic patterns
    base_amount = np.random.lognormal(mean=10, sigma=1)  # Base amount in CLP
    
    # Occasionally generate very large amounts (1% chance)
    if random.random() < 0.01:
        base_amount *= 10
    
    amount = base_amount
    
    # Number of failed attempts before success
    # Most transactions succeed on first try
    failed_attempts = np.random.geometric(p=0.9) - 1
    
    return {
        'transaction_id': transaction_id,
        'amount': round(amount, 2),
        'failed_attempts': failed_attempts
    }

def generate_fake_transactions(num_transactions=10000, start_date=None):
    """
    Generate fake transaction data with auto-incremented IDs, amount and failed attempts.
    
    Args:
        num_transactions (int): Number of transactions to generate
        start_date (datetime): Start date for transactions
    
    Returns:
        pandas.DataFrame: Generated transaction data
    """
    transactions = []
    for i in range(1, num_transactions + 1):
        transaction = generate_single_transaction(i)
        transactions.append(transaction)
    
    return pd.DataFrame(transactions)

def save_fake_data(data, filename='data/generated_data.csv'):
    """
    Save generated transaction data to a CSV file.
    
    Args:
        data (pandas.DataFrame): The transaction data to save
        filename (str): Path where to save the CSV file
    """
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save to CSV
    data.to_csv(filename, index=False)
    print(f"Datos de muestra generados exitosamente y guardados en {filename}")

if __name__ == "__main__":
    # Generate and save sample data
    data = generate_fake_transactions()
    save_fake_data(data) 