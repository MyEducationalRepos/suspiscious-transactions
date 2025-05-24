"""
Module for the Streamlit-based user interface of the application.
"""
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.anomaly_detector import initialize_isolation_forest, train_model, predict_anomalies
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
    
    # Ensure all features are numeric
    for feature in numeric_features:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')
    
    # Fill any NaN values with 0
    df[numeric_features] = df[numeric_features].fillna(0)
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[numeric_features])
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    
    # Create DataFrame with PCA results
    pca_df = pd.DataFrame(
        data=pca_result,
        columns=['PC1', 'PC2']
    )
    
    # Add original data for reference (only numeric columns)
    pca_df['monto'] = df['monto']
    pca_df['fecha'] = df['fecha']
    pca_df['es_anomalia'] = df['es_anomalia'].astype(bool)  # Convert to boolean
    pca_df['tipo_transaccion'] = df['tipo_transaccion'].astype(str)  # Convert to string
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    
    return pca_df, explained_variance, pca.components_, numeric_features

def explain_anomaly(row):
    """
    Explain why a transaction is marked as suspicious.
    """
    reasons = []
    
    # Check amount-based risk
    if row['riesgo_monto'] > 2:  # More than 2 standard deviations
        reasons.append(f"Monto inusual: ${row['monto']:,.2f} CLP")
    
    # Check time-based risk
    if row['riesgo_tiempo'] > 0.7:  # High time risk
        hour = pd.to_datetime(row['fecha']).hour
        reasons.append(f"Hora inusual: {hour}:00")
    
    # Check location risk
    if row['riesgo_ubicacion'] < 0.1:  # Rare location
        reasons.append(f"Ubicación inusual: {row['ubicacion']}")
    
    # Check weekend risk
    if row['riesgo_fin_semana'] > 0:
        reasons.append("Transacción en fin de semana")
    
    # Check security risk
    if row['riesgo_seguridad'] > 0.7:
        reasons.append(f"Alto riesgo de seguridad (Dispositivo: {row['puntaje_confianza_dispositivo']:.1f}, IP: {row['puntaje_riesgo_ip']:.1f})")
    
    # Check behavior risk
    if row['riesgo_conducta'] > 0.5:
        reasons.append(f"Conducta sospechosa (Intentos fallidos: {row['intentos_fallidos']}, Transacciones recientes: {row['transacciones_recientes_24h']})")
    
    # Check transaction type
    if row['tipo_transaccion'] in ['retiro', 'transferencia']:
        reasons.append(f"Tipo de transacción de alto riesgo: {row['tipo_transaccion']}")
    
    return " | ".join(reasons) if reasons else "Múltiples factores"

def main(data=None):
    """
    Main function to run the Streamlit UI.
    
    Args:
        data (pandas.DataFrame, optional): Pre-loaded transaction data
    """
    # Set page title
    st.set_page_config(page_title="Detector de Anomalías en Transacciones", layout="wide")
    
    # Display main title
    st.title("Detector de Anomalías en Transacciones")
    
    # Display description
    st.markdown("""
    Esta aplicación ayuda a detectar transacciones sospechosas usando aprendizaje automático.
    Analiza múltiples factores incluyendo:
    - Montos y patrones de transacciones
    - Tiempo entre transacciones
    - Patrones de ubicación
    - Tipos de transacciones
    - Patrones de fin de semana vs días laborables
    - Cambios de ubicación
    - Cambios en tipos de transacciones
    - Métricas de seguridad (confianza del dispositivo, riesgo de IP)
    - Patrones de conducta (intentos fallidos, transacciones recientes)
    """)
    
    # Add file uploader
    uploaded_file = st.file_uploader(
        "Sube tus datos de transacciones (CSV o Excel)",
        type=['csv', 'xlsx'],
        help="Sube un archivo CSV o Excel con datos de transacciones"
    )
    
    # Display data if available
    if data is not None:
        st.subheader("Datos de Transacciones Actuales")
        st.dataframe(data.head())
        st.write(f"Total de transacciones: {len(data)}")
        
        # Add anomaly detection button
        if st.button("Detectar Anomalías"):
            with st.spinner("Analizando transacciones..."):
                # Initialize and train the model
                model = initialize_isolation_forest()
                trained_model = train_model(model, data)
                
                # Get predictions
                results = predict_anomalies(trained_model, data)
                
                # Display results
                st.subheader("Resultados de Detección de Anomalías")
                
                # Show summary
                num_anomalies = results['es_anomalia'].sum()
                st.write(f"Se encontraron {num_anomalies} transacciones sospechosas de {len(results)} transacciones totales")
                
                # Create visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Visualización PCA de Transacciones")
                    # Prepare PCA features
                    pca_df, explained_variance, components, numeric_features = prepare_pca_features(results)
                    
                    # Create PCA plot
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    
                    # Plot transactions by type
                    for tipo in pca_df['tipo_transaccion'].unique():
                        subset = pca_df[pca_df['tipo_transaccion'] == tipo]
                        sns.scatterplot(data=subset, x='PC1', y='PC2', 
                                      label=tipo, alpha=0.6, ax=ax1)
                    
                    # Highlight anomalies
                    anomalies = pca_df[pca_df['es_anomalia']]
                    sns.scatterplot(data=anomalies, x='PC1', y='PC2', 
                                  color='red', marker='x', s=100, 
                                  label='Sospechoso', alpha=0.8, ax=ax1)
                    
                    # Add explained variance to axis labels
                    plt.xlabel(f'PC1 ({explained_variance[0]:.1%} varianza)')
                    plt.ylabel(f'PC2 ({explained_variance[1]:.1%} varianza)')
                    plt.title('Patrones de Transacciones (PCA)')
                    plt.legend(title='Tipo de Transacción')
                    st.pyplot(fig1)
                    
                    # Show feature contributions to PCs
                    st.write("Contribución de Características a los Componentes Principales:")
                    feature_names = [
                        'Monto', 'Duración', 'Transacciones Recientes',
                        'Saldo', 'Comisión', 'Distancia',
                        'Intentos Fallidos', 'Confianza Dispositivo', 'Riesgo IP'
                    ]
                    pc_contributions = pd.DataFrame(
                        components.T,
                        columns=['PC1', 'PC2'],
                        index=feature_names
                    )
                    st.dataframe(pc_contributions.style.background_gradient(cmap='RdBu'))
                
                with col2:
                    st.subheader("Distribución de Factores de Riesgo")
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    
                    # Plot risk factors
                    risk_data = pd.melt(results[['riesgo_monto', 'riesgo_tiempo', 'riesgo_ubicacion', 
                                               'riesgo_fin_semana', 'riesgo_seguridad', 'riesgo_conducta', 
                                               'es_anomalia']], 
                                      id_vars=['es_anomalia'],
                                      var_name='Factor de Riesgo',
                                      value_name='Puntaje de Riesgo')
                    
                    sns.boxplot(data=risk_data, x='Factor de Riesgo', y='Puntaje de Riesgo', 
                              hue='es_anomalia', ax=ax2)
                    
                    plt.title('Distribución de Factores de Riesgo')
                    plt.xticks(rotation=45)
                    plt.legend(title='Estado de Transacción')
                    st.pyplot(fig2)
                
                # Display detailed results
                st.subheader("Análisis de Transacciones Sospechosas")
                
                # Add explanation for each suspicious transaction
                anomalies = results[results['es_anomalia']]
                anomalies['razon'] = anomalies.apply(explain_anomaly, axis=1)
                
                # Show risk factors for suspicious transactions
                st.write("Factores de Riesgo para Transacciones Sospechosas:")
                risk_factors = anomalies[['monto', 'riesgo_monto', 'riesgo_tiempo', 'riesgo_ubicacion', 
                                       'riesgo_fin_semana', 'riesgo_seguridad', 'riesgo_conducta', 
                                       'puntaje_anomalia', 'razon']]
                st.dataframe(risk_factors.style.background_gradient(cmap='Reds'))
                
                # Show full details of suspicious transactions
                st.write("Detalles Completos de Transacciones Sospechosas:")
                st.dataframe(anomalies)
                
                # Show normal transactions
                st.subheader("Transacciones Normales")
                st.dataframe(results[~results['es_anomalia']])

if __name__ == "__main__":
    main() 