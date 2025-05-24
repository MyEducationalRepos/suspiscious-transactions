"""
Script to generate sample transaction data.
This should be run before starting the Streamlit application.
"""
from src.data_generator import generate_fake_transactions, save_fake_data

def main():
    print("Generando datos de transacciones de muestra...")
    data = generate_fake_transactions(10000)  # Generate 10000 sample transactions
    save_fake_data(data)
    print("Datos generados exitosamente. Ahora puede ejecutar la aplicación Streamlit.")

if __name__ == "__main__":
    main() 