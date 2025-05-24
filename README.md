# Detección de Anomalías en Transacciones

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Esta aplicación utiliza Isolation Forest para detectar transacciones sospechosas en tiempo real. El sistema analiza múltiples factores como monto, tiempo, ubicación y tipo de transacción para identificar patrones inusuales.

## Teoría de Isolation Forest

### Concepto Básico
Isolation Forest es un algoritmo de detección de anomalías que se basa en el principio de que las anomalías son más fáciles de "aislar" que los puntos normales. El algoritmo funciona de la siguiente manera:

1. **Aislamiento**: 
   - Selecciona aleatoriamente un atributo
   - Selecciona aleatoriamente un valor entre el máximo y mínimo del atributo
   - Divide los datos en dos grupos basados en este valor
   - Repite el proceso recursivamente

2. **Medida de Anomalía**:
   - Las anomalías requieren menos divisiones para ser aisladas
   - La longitud de la ruta desde la raíz hasta el nodo hoja es más corta para anomalías
   - Se calcula un score de anomalía basado en esta longitud

### Ventajas del Isolation Forest
- Eficiente en memoria y tiempo de cómputo
- Funciona bien con datos de alta dimensionalidad
- No requiere etiquetado previo de anomalías
- Robusto a ruido en los datos

### Parámetros Clave
- `n_estimators`: Número de árboles en el bosque (100 en nuestra implementación)
- `contamination`: Proporción esperada de anomalías (0.014 para ~14 anomalías en 1000 transacciones)
- `max_samples`: Tamaño de la submuestra para entrenar cada árbol

## Características del Sistema

### 1. Generación de Datos
- Simulación de transacciones bancarias realistas
- Patrones temporales (más transacciones en horario laboral)
- Distribuciones realistas de montos y ubicaciones
- Características incluidas:
  - Monto
  - Tipo de transacción
  - Ubicación
  - Duración
  - Transacciones recientes
  - Saldo anterior
  - Comisión
  - Distancia
  - Intentos fallidos

### 2. Detección de Anomalías
- Preprocesamiento de características:
  - Codificación one-hot para variables categóricas
  - Escalado de características numéricas
  - Cálculo de estadísticas móviles
- Visualización PCA para reducción de dimensionalidad
- Explicación de anomalías basada en múltiples factores

### 3. Visualización
- Gráfico PCA de transacciones normales vs. anómalas
- Distribución de montos por tipo de transacción
- Tabla detallada de transacciones sospechosas
- Contribución de características al PCA

## Requisitos

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Faker

## Instalación

1. Clonar el repositorio:
```bash
git clone <repository-url>
cd suspicious-transactions
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

1. Generar datos de muestra:
```bash
python generate_data.py
```
Este script generará 1000 transacciones de muestra y las guardará en `data/generated_data.csv`.

2. Iniciar la aplicación:
```bash
streamlit run main.py
```
La aplicación estará disponible en http://localhost:8501

## Estructura del Proyecto

```
.
├── data/
│   └── generated_data.csv    # Datos generados
├── src/
│   ├── data_generator.py     # Generación de datos sintéticos
│   ├── anomaly_detector.py   # Implementación de Isolation Forest
│   └── ui_new.py            # Interfaz de usuario Streamlit
├── main.py                  # Punto de entrada principal
├── generate_data.py         # Script de generación de datos
└── requirements.txt         # Dependencias del proyecto
```

## Explicación de Anomalías

El sistema identifica transacciones sospechosas basándose en:

1. **Monto Inusual**:
   - Transacciones significativamente mayores al promedio
   - Desviación estándar > 2 del promedio

2. **Patrones Temporales**:
   - Transacciones fuera de horario (antes de 6 AM o después de 10 PM)
   - Actividad inusual en fin de semana

3. **Ubicación**:
   - Distancias inusuales entre transacciones
   - Cambios de ubicación sospechosos

4. **Comportamiento**:
   - Múltiples intentos fallidos
   - Patrones de transacción inusuales

## Contribución

Las contribuciones son bienvenidas. Por favor, asegúrese de:
1. Hacer fork del repositorio
2. Crear una rama para su feature
3. Commit sus cambios
4. Push a la rama
5. Crear un Pull Request

## Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

La Licencia MIT es una licencia permisiva que permite:
- Uso comercial
- Modificación
- Distribución
- Uso privado

Con las siguientes condiciones:
- Incluir la notificación de copyright original
- Incluir la licencia MIT en todas las copias o partes sustanciales del software

Para más información sobre la Licencia MIT, visite [opensource.org/licenses/MIT](https://opensource.org/licenses/MIT).
