# Intrusion Detection System — Streamlit app (NSL-KDD)

Classification-based intrusion detection with three models
(Decision Tree, Random Forest, Naive Bayes) and an interactive interface.

## Folder layout

```
ids_streamlit/
├── app.py              # Streamlit interface
├── ids_core.py         # data loading, training, prediction helpers
├── requirements.txt
├── .streamlit/config.toml   # theme (colours, fonts)
├── KDDTrain_.arff      # <- copy your NSL-KDD files here
└── KDDTest_.arff
```

## Run locally

```bash
# 1. (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. install the dependencies
pip install -r requirements.txt

# 3. start the app
streamlit run app.py
```

The browser opens at http://localhost:8501.

The first start trains the three models (about a minute) and saves them to
`ids_model_cache.joblib`; later starts load that file and open instantly.
Delete the file if you change the data or the model settings.

## What is in the app

* **Predict** – load a real record from the test set (or type values by hand) and classify it.
  All three models are shown side by side, with the actual label when a real record is loaded.
* **Model comparison** – accuracy / precision / recall / F1, confusion matrices, classification reports.
* **Interface validation** – actual vs predicted for chosen test rows (default 411, 0, 50, 100, 250, 500, 2000).
* **Dataset** – record counts, duplicates, class distribution, sample rows.

## Expected results (KDDTest+, 22,541 records after removing duplicates)

| Model | Accuracy |
|---|---|
| Decision Tree | 77.22 % |
| Random Forest | 77.97 % |
| Naive Bayes | 45.04 % |
