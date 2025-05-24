# MVP Build Plan: Transaction Anomaly Detection Tool

## Task List

1.  **Setup Project Directory Structure**
    *   **Goal:** Create the basic folder structure (`data/`, `src/`) and essential files (`main.py`, `requirements.txt`, `README.md`).
    *   **Start:** Project directory is empty.
    *   **End:** Directories and empty files are created as per the architecture.
    *   **Test:** Verify the presence of the created directories and files.

2.  **Create `requirements.txt`**
    *   **Goal:** List the initial required Python libraries (`faker`, `pandas`, `scikit-learn`, `seaborn`, `openpyxl`, `streamlit` or chosen UI library).
    *   **Start:** `requirements.txt` is empty.
    *   **End:** `requirements.txt` contains the list of core dependencies.
    *   **Test:** Open `requirements.txt` and confirm the listed libraries.

3.  **Implement Basic Data Generation (`data_generator.py`) - Setup**
    *   **Goal:** Create the `data_generator.py` file and import necessary libraries (`faker`, `pandas`).
    *   **Start:** `src/data_generator.py` does not exist or is empty.
    *   **End:** `src/data_generator.py` exists and has basic imports.
    *   **Test:** Verify the file exists and contains import statements.

4.  **Implement Basic Data Generation (`data_generator.py`) - Faker Instance**
    *   **Goal:** Initialize a `Faker` instance with the Chilean Spanish locale (`es_CL`).
    *   **Start:** `src/data_generator.py` has imports but no Faker instance.
    *   **End:** `src/data_generator.py` initializes `Faker(locale='es_CL')`.
    *   **Test:** Add a print statement to confirm the instance is created (will be removed later).

5.  **Implement Basic Data Generation (`data_generator.py`) - Generate Single Record**
    *   **Goal:** Write code to generate a single, simple transaction record using the Faker instance (e.g., transaction ID, amount, date).
    *   **Start:** `src/data_generator.py` has a Faker instance but no data generation logic.
    *   **End:** `src/data_generator.py` has a function or code block that generates one dictionary representing a transaction.
    *   **Test:** Call the function/block and print the generated dictionary.

6.  **Implement Basic Data Generation (`data_generator.py`) - Generate Multiple Records**
    *   **Goal:** Write a function `generate_fake_transactions(num_records)` that generates a list of multiple transaction records.
    *   **Start:** `src/data_generator.py` can generate only one record.
    *   **End:** `src/data_generator.py` has a function that generates a list of `num_records` dictionaries.
    *   **Test:** Call `generate_fake_transactions(5)` and print the list to verify multiple records are generated.

7.  **Implement Basic Data Generation (`data_generator.py`) - Create DataFrame**
    *   **Goal:** Convert the list of generated records into a pandas DataFrame.
    *   **Start:** `generate_fake_transactions` returns a list of dictionaries.
    *   **End:** `generate_fake_transactions` returns a pandas DataFrame.
    *   **Test:** Call the function and print the type and head of the returned object to confirm it's a DataFrame.

8.  **Implement Basic Data Generation (`data_generator.py`) - Save to CSV**
    *   **Goal:** Add functionality to save the generated DataFrame to `data/generated_data.csv`.
    *   **Start:** `generate_fake_transactions` returns a DataFrame but doesn't save it.
    *   **End:** `generate_generator.py` has a function `save_transactions_to_csv(df, filename)` that saves the DataFrame.
    *   **Test:** Call the save function and verify that `data/generated_data.csv` is created and contains data.

9.  **Implement Basic Data Processing (`data_processor.py`) - Setup**
    *   **Goal:** Create the `data_processor.py` file and import necessary libraries (`pandas`).
    *   **Start:** `src/data_processor.py` does not exist or is empty.
    *   **End:** `src/data_processor.py` exists and has basic imports.
    *   **Test:** Verify the file exists and contains import statements.

10. **Implement Basic Data Processing (`data_processor.py`) - Load CSV**
    *   **Goal:** Write a function `load_data_from_csv(filename)` that loads data from `data/generated_data.csv` into a pandas DataFrame.
    *   **Start:** `src/data_processor.py` has imports but no loading function.
    *   **End:** `src/data_processor.py` has a function that reads a CSV file into a DataFrame.
    *   **Test:** Call the function with the generated CSV file path and print the head of the returned DataFrame.

11. **Implement Basic Anomaly Detection (`anomaly_detector.py`) - Setup**
    *   **Goal:** Create the `anomaly_detector.py` file and import necessary libraries (`sklearn.ensemble.IsolationForest`).
    *   **Start:** `src/anomaly_detector.py` does not exist or is empty.
    *   **End:** `src/anomaly_detector.py` exists and has basic imports.
    *   **Test:** Verify the file exists and contains import statements.

12. **Implement Basic Anomaly Detection (`anomaly_detector.py`) - Initialize Model**
    *   **Goal:** Write a function `initialize_isolation_forest()` that returns an initialized `IsolationForest` model instance.
    *   **Start:** `src/anomaly_detector.py` has imports but no model initialization.
    *   **End:** `src/anomaly_detector.py` has a function that creates and returns an `IsolationForest` object.
    *   **Test:** Call the function and print the type of the returned object.

13. **Implement Basic Anomaly Detection (`anomaly_detector.py`) - Train Model**
    *   **Goal:** Write a function `train_model(model, data)` that trains the Isolation Forest model on a given pandas DataFrame.
    *   **Start:** `src/anomaly_detector.py` can initialize a model but not train it.
    *   **End:** `src/anomaly_detector.py` has a function that fits the model to data.
    *   **Test:** Generate some simple numerical data (e.g., a small DataFrame with one column), initialize a model, call `train_model`, and check if the model object is modified (e.g., has fitted attributes).

14. **Implement Basic Anomaly Detection (`anomaly_detector.py`) - Predict Anomalies**
    *   **Goal:** Write a function `predict_anomalies(model, data)` that uses the trained model to predict anomaly labels or scores for the data.
    *   **Start:** `src/anomaly_detector.py` can train a model but not predict.
    *   **End:** `src/anomaly_detector.py` has a function that calls `model.predict()` or `model.decision_function()` and returns the results.
    *   **Test:** Train a model on simple data, call `predict_anomalies`, and print the output to see the predictions (e.g., -1 for outliers, 1 for inliers).

15. **Implement Basic UI (`ui.py`) - Setup**
    *   **Goal:** Create the `ui.py` file and import the chosen UI library (e.g., `streamlit`).
    *   **Start:** `src/ui.py` does not exist or is empty.
    *   **End:** `src/ui.py` exists and has basic imports for the UI library.
    *   **Test:** Verify the file exists and contains import statements.

16. **Implement Basic UI (`ui.py`) - Display Title**
    *   **Goal:** Add code to display a simple title for the application in the UI.
    *   **Start:** `src/ui.py` has imports but no UI elements.
    *   **End:** `src/ui.py` displays a title like "Transaction Anomaly Detector".
    *   **Test:** Run the UI script (e.g., `streamlit run src/ui.py`) and verify the title is displayed in the browser.

17. **Implement Basic UI (`ui.py`) - File Uploader Placeholder**
    *   **Goal:** Add a placeholder UI element for uploading a file (e.g., `st.file_uploader` in Streamlit).
    *   **Start:** `src/ui.py` only displays a title.
    *   **End:** `src/ui.py` displays a file upload widget.
    *   **Test:** Run the UI script and verify the file uploader appears.

18. **Connect Main Script (`main.py`) - Setup**
    *   **Goal:** Create the `main.py` file and import necessary modules from `src`.
    *   **Start:** `main.py` does not exist or is empty.
    *   **End:** `main.py` exists and imports modules like `data_generator`, `data_processor`, etc.
    *   **Test:** Verify the file exists and contains import statements.

19. **Connect Main Script (`main.py`) - Generate Data on Run**
    *   **Goal:** Modify `main.py` to call the data generation and saving functions from `data_generator.py` when executed directly.
    *   **Start:** `main.py` has imports but no execution logic.
    *   **End:** `main.py` calls `generate_fake_transactions` and `save_transactions_to_csv`.
    *   **Test:** Run `python main.py` and verify that `data/generated_data.csv` is created/updated.

20. **Connect Main Script (`main.py`) - Load Data and Run UI**
    *   **Goal:** Modify `main.py` to load the generated data using `data_processor.py` and then launch the UI from `ui.py`, potentially passing the loaded data.
    *   **Start:** `main.py` only generates data.
    *   **End:** `main.py` loads data and calls a function in `ui.py` to start the application, passing the data.
    *   **Test:** Run `python main.py` (or the command to run the UI, e.g., `streamlit run main.py` if the UI entry point is in main) and verify the UI starts and potentially shows confirmation of data loading.

