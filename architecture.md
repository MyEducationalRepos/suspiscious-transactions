# Project Architecture: Transaction Anomaly Detection Tool

This document outlines the architecture for a Python-based tool designed to demonstrate anomaly and outlier detection in financial transactions for compliance students.

## File and Folder Structure


. ├── data/ │ └── generated_data.csv ├── src/ │ ├── init.py │ ├── data_generator.py │ ├── data_processor.py │ ├── anomaly_detector.py │ ├── visualizer.py │ └── ui.py ├── main.py ├── requirements.txt └── README.md


## What Each Part Does

*   **`data/`**: This directory will store data files.
    *   **`generated_data.csv`**: A CSV file containing the synthetic transaction data generated by the tool.
*   **`src/`**: This directory contains the core Python modules of the application.
    *   **`__init__.py`**: An empty file that indicates the `src` directory is a Python package.
    *   **`data_generator.py`**:
        *   Responsible for creating synthetic transaction data.
        *   Uses the `faker` library with a Chilean Spanish locale (`es_CL`) to generate realistic-looking data points such as transaction IDs, dates, amounts, account numbers, transaction types, locations, etc.
        *   Saves the generated data into a CSV file (`data/generated_data.csv`).
    *   **`data_processor.py`**:
        *   Handles data loading and initial processing.
        *   Can load data from the generated CSV file or an uploaded Excel file using the `pandas` library.
        *   Performs necessary data cleaning, transformation, and feature engineering steps required for anomaly detection.
        *   Provides functions to access and manipulate the loaded data (e.g., selecting relevant columns, handling missing values).
    *   **`anomaly_detector.py`**:
        *   Implements the anomaly detection logic.
        *   Uses the `scikit-learn` library, specifically the `IsolationForest` algorithm, to identify outliers and potentially suspicious transactions based on the processed data.
        *   Provides functions to train the model and predict anomaly scores or labels for new data.
    *   **`visualizer.py`**:
        *   Responsible for creating data visualizations.
        *   Uses the `seaborn` library (which builds on Matplotlib) to generate plots such as histograms of transaction amounts, scatter plots of features, or visualizations highlighting detected anomalies.
        *   Helps in understanding the data distribution and the results of the anomaly detection.
    *   **`ui.py`**:
        *   Contains the code for the simple user interface.
        *   Could be built using a library like `Streamlit` for rapid development of interactive web applications, or a desktop GUI library like `Tkinter` or `PyQt`.
        *   Allows users to upload an Excel file, trigger data processing and anomaly detection, and display the results (tables, visualizations).
*   **`main.py`**:
    *   The main entry point of the application.
    *   Orchestrates the flow by calling functions from the modules in the `src` directory.
    *   Initializes the UI and handles user interactions by invoking the appropriate data processing, detection, and visualization steps.
*   **`requirements.txt`**:
    *   Lists all the necessary Python libraries required to run the project (e.g., `faker`, `pandas`, `seaborn`, `scikit-learn`, `streamlit` or other UI library, `openpyxl` for reading Excel).
*   **`README.md`**:
    *   Provides a description of the project, setup instructions (how to install dependencies), how to run the application, and basic usage guidelines.

## Where State Lives and How Services Connect

*   **State Management**:
    *   The primary data state (the transaction data) is managed within `data_processor.py`, typically held in `pandas` DataFrames. This DataFrame is passed between modules as needed.
    *   The trained `IsolationForest` model is an object instance managed within `anomaly_detector.py`.
    *   The state of the user interface (e.g., uploaded file, current view, displayed results) is managed within `ui.py`, according to the conventions of the chosen UI library.
*   **Service Connections**:
    *   **`main.py`** acts as the central orchestrator. It imports and calls functions/classes from the modules in `src/`.
    *   **`data_generator.py`** operates somewhat independently, writing data to the file system.
    *   **`ui.py`** interacts directly with the user and calls functions in `data_processor.py`, `anomaly_detector.py`, and `visualizer.py` based on user actions (e.g., clicking a button to upload data, clicking a button to run analysis).
    *   **`data_processor.py`** reads data from the file system (CSV/Excel) and passes the processed `pandas` DataFrame to `anomaly_detector.py` and `visualizer.py`.
    *   **`anomaly_detector.py`** receives data from `data_processor.py`, performs calculations, and returns results (like anomaly scores or labels) back to the caller, typically `ui.py` or `data_processor.py`.
    *   **`visualizer.py`** receives data and results from `data_processor.py` or `anomaly_detector.py` and generates plots, which are then displayed by `ui.py`.
    *   Communication between modules primarily happens through function calls and passing data (like Pandas DataFrames) as arguments. There isn't a complex service-oriented architecture;