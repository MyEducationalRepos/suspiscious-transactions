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
    numeric_features = ['amount', 'failed_attempts']
    
    # Create numeric data for PCA
    numeric_data = df[numeric_features].copy()
    
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
    pca_df['amount'] = df['amount']
    pca_df['failed_attempts'] = df['failed_attempts']
    pca_df['is_anomaly'] = df['is_anomaly']
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    
    return {
        'pca_df': pca_df,
        'varianza_explicada': explained_variance,
        'componentes': pd.DataFrame(
            pca.components_,
            columns=numeric_features
        )
    }

def main(data):
    """
    Main function to run the Streamlit UI.
    
    Args:
        data (pandas.DataFrame): Transaction data
    """
    st.title("Detección de Anomalías en Transacciones")
    
    st.write("""
    Esta aplicación utiliza Isolation Forest para detectar transacciones sospechosas basándose en:
    - Monto de la transacción
    - Número de intentos fallidos
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
            data = pd.read_csv(uploaded_file)
            st.session_state.show_dataframes = True
        except Exception as e:
            st.error(f"Error al cargar el archivo: {str(e)}")
            st.stop()
    
    # Train model and get predictions
    model = initialize_isolation_forest()
    trained_model = train_model(model, data)
    results = predict_anomalies(trained_model, data)
    
    # Get anomalies
    anomalies = results[results['is_anomaly']]
    
    # Calculate global statistics
    global_stats = {
        'amount_mean': results['amount'].mean(),
        'amount_std': results['amount'].std(),
        'failed_attempts_mean': results['failed_attempts'].mean(),
        'failed_attempts_std': results['failed_attempts'].std()
    }
    
    # Prepare features for PCA
    pca_results = prepare_pca_features(results)
    
    # Create two columns for the plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Visualización PCA de Transacciones")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot normal transactions
        normal_mask = ~results['is_anomaly']
        ax.scatter(
            pca_results['pca_df'].loc[normal_mask, 'PC1'],
            pca_results['pca_df'].loc[normal_mask, 'PC2'],
            c='blue',
            alpha=0.5,
            label='Normal'
        )
        
        # Plot anomalies
        anomaly_mask = results['is_anomaly']
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
        
        sns.boxplot(data=results, x='is_anomaly', y='amount', ax=ax)
        ax.set_xlabel('Es Anomalía')
        ax.set_ylabel('Monto')
        st.pyplot(fig)
    
    # Show anomalies table only if show_dataframes is True
    if st.session_state.show_dataframes:
        st.subheader("Transacciones Sospechosas Detectadas")
        
        # Format the anomalies table
        anomalies_display = anomalies[[
            'transaction_id', 'amount', 'failed_attempts', 'anomaly_score'
        ]].copy()
        
        # Format numeric columns
        for col in ['amount', 'anomaly_score']:
            anomalies_display[col] = anomalies_display[col].round(2)
        
        # Display the table
        st.dataframe(anomalies_display)
        
        # Show feature contributions to PCA
        st.subheader("Contribución de Características al PCA")
        feature_contributions = pca_results['componentes']
        st.dataframe(feature_contributions.style.background_gradient(cmap='RdYlBu_r')) 