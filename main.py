"""
Main module for the transaction anomaly detection application.
"""
import streamlit as st
import pandas as pd
from src.ui_new import main as ui_main

if __name__ == "__main__":
    # Load the pre-generated data
    try:
        data = pd.read_csv('data/generated_data.csv', encoding='utf-8', quoting=1)
        data['fecha'] = pd.to_datetime(data['fecha'])
        ui_main(data)
    except FileNotFoundError:
        st.error("No se encontró el archivo de datos. Por favor, ejecute primero el script de generación de datos.")
        st.stop()
