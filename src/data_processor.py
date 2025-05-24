"""
Module for processing and preparing transaction data for analysis.
"""
import pandas as pd

def load_data_from_csv(filename='data/generated_data.csv'):
    """
    Load transaction data from a CSV file.
    
    Args:
        filename (str): Path to the CSV file (default: 'data/generated_data.csv')
        
    Returns:
        pandas.DataFrame: The loaded transaction data
        
    Raises:
        FileNotFoundError: If the specified file does not exist
    """
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the file: {filename}") 