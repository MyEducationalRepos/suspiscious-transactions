import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.anomaly_detector import initialize_isolation_forest, train_model, predict_anomalies

def prepare_pca_features(data):
    """
    Prepare features for PCA visualization.
    """
    # Create a copy of the data
    df = data.copy()
    
    # Select only numeric features for PCA
    numeric_features = [
        'monto', 'duracion_segundos', 'transacciones_recientes_24h',
        'saldo_anterior', 'comision', 'distancia_km',
        'intentos_fallidos', 'puntaje_confianza_dispositivo', 'puntaje_riesgo_ip'
    ]
    
    # Filter only the numeric features that exist in the dataframe and are actually numeric
    available_numeric_features = []
    for feature in numeric_features:
        if feature in df.columns:
            try:
                # Try to convert to numeric, if successful, add to features
                df[feature] = pd.to_numeric(df[feature], errors='raise')
                available_numeric_features.append(feature)
            except Exception as e:
                st.write(f"Error converting {feature} to numeric: {str(e)}")
                continue
    
    if not available_numeric_features:
        raise ValueError("No numeric features available for PCA")
    
    # Create numeric data for PCA
    numeric_data = df[available_numeric_features].copy()
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(numeric_data)
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    
    # Create DataFrame with PCA results
    pca_df = pd.DataFrame(
        data=pca_result,
        columns=['PC1', 'PC2']
    )
    
    # Add original data for reference
    pca_df['monto'] = df['monto']
    pca_df['fecha'] = df['fecha']
    pca_df['es_anomalia'] = df['es_anomalia']
    pca_df['tipo_transaccion'] = df['tipo_transaccion']
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    
    return {
        'pca_df': pca_df,
        'varianza_explicada': explained_variance,
        'componentes': pd.DataFrame(
            pca.components_,
            columns=available_numeric_features
        )
    }

def explain_anomaly(row, global_stats):
    """
    Explain why a transaction is marked as suspicious.
    
    Args:
        row (pandas.Series): The transaction row to analyze
        global_stats (dict): Dictionary containing global statistics for comparison
    """
    reasons = []
    
    # Check amount
    if row['monto'] > global_stats['monto_mean'] + 2 * global_stats['monto_std']:
        reasons.append("Monto inusual")
    
    # Check time
    if row['fecha'].hour < 6 or row['fecha'].hour > 22:
        reasons.append("Hora inusual")
    
    # Check location
    if row['distancia_km'] > global_stats['distancia_mean'] + 2 * global_stats['distancia_std']:
        reasons.append("Distancia inusual")
    
    # Check transaction type
    if row['tipo_transaccion'] in ['retiro', 'transferencia'] and row['monto'] > global_stats['monto_mean']:
        reasons.append("Tipo de transacción inusual")
    
    # Check failed attempts
    if row['intentos_fallidos'] > 2:
        reasons.append("Múltiples intentos fallidos")
    
    return " | ".join(reasons) if reasons else "Múltiples factores"

def main(data):
    """
    Main function to run the Streamlit UI.
    
    Args:
        data (pandas.DataFrame): Transaction data
    """
    st.title("Detección de Anomalías en Transacciones")
    
    st.write("""
    Esta aplicación utiliza Isolation Forest para detectar transacciones sospechosas basándose en múltiples factores:
    - Monto de la transacción
    - Patrones de tiempo (hora del día, día de la semana)
    - Ubicación y distancia
    - Tipo de transacción
    - Comportamiento del usuario
    """)
    
    # Initialize session state for dataframes if not exists
    if 'show_dataframes' not in st.session_state:
        st.session_state.show_dataframes = False
    
    # Add a refresh button
    if st.button("Actualizar Datos"):
        st.session_state.show_dataframes = True
    
    # File uploader
    uploaded_file = st.file_uploader("Subir archivo CSV con transacciones", type=['csv'])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file, encoding='utf-8', quoting=1)
            data['fecha'] = pd.to_datetime(data['fecha'])
            st.session_state.show_dataframes = True
        except Exception as e:
            st.error(f"Error al cargar el archivo: {str(e)}")
            st.stop()
    
    # Train model and get predictions
    model = initialize_isolation_forest()
    trained_model = train_model(model, data)
    results = predict_anomalies(trained_model, data)
    
    # Get anomalies
    anomalies = results[results['es_anomalia']]
    
    # Calculate global statistics
    global_stats = {
        'monto_mean': results['monto'].mean(),
        'monto_std': results['monto'].std(),
        'distancia_mean': results['distancia_km'].mean(),
        'distancia_std': results['distancia_km'].std()
    }
    
    # Prepare features for PCA
    pca_results = prepare_pca_features(results)
    
    # Create two columns for the plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Visualización PCA de Transacciones")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot normal transactions
        normal_mask = ~results['es_anomalia']
        ax.scatter(
            pca_results['pca_df'].loc[normal_mask, 'PC1'],
            pca_results['pca_df'].loc[normal_mask, 'PC2'],
            c='blue',
            alpha=0.5,
            label='Normal'
        )
        
        # Plot anomalies
        anomaly_mask = results['es_anomalia']
        ax.scatter(
            pca_results['pca_df'].loc[anomaly_mask, 'PC1'],
            pca_results['pca_df'].loc[anomaly_mask, 'PC2'],
            c='red',
            alpha=0.7,
            label='Anomalía'
        )
        
        ax.set_xlabel(f'PC1 ({pca_results["varianza_explicada"][0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca_results["varianza_explicada"][1]:.1%})')
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Distribución de Montos")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.boxplot(data=results, x='tipo_transaccion', y='monto', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_xlabel('Tipo de Transacción')
        ax.set_ylabel('Monto')
        st.pyplot(fig)
    
    # Show anomalies table only if show_dataframes is True
    if st.session_state.show_dataframes:
        st.subheader("Transacciones Sospechosas Detectadas")
        
        # Format the anomalies table
        anomalies_display = anomalies[[
            'fecha', 'monto', 'tipo_transaccion', 'ubicacion',
            'duracion_segundos', 'transacciones_recientes_24h',
            'saldo_anterior', 'comision', 'distancia_km',
            'intentos_fallidos', 'puntaje_anomalia'
        ]].copy()
        
        # Add explanation column using global statistics
        anomalies_display['razon'] = anomalies_display.apply(
            lambda row: explain_anomaly(row, global_stats),
            axis=1
        )
        
        # Format numeric columns
        for col in ['monto', 'duracion_segundos', 'transacciones_recientes_24h',
                   'saldo_anterior', 'comision', 'distancia_km',
                   'intentos_fallidos', 'puntaje_anomalia']:
            anomalies_display[col] = anomalies_display[col].round(2)
        
        # Display the table
        st.dataframe(anomalies_display)
        
        # Show feature contributions to PCA
        st.subheader("Contribución de Características al PCA")
        feature_contributions = pca_results['componentes']
        st.dataframe(feature_contributions.style.background_gradient(cmap='RdYlBu_r')) 