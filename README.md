# 🛡️ Intrusion Detection System (NSL-KDD)

A data mining project that classifies network connections as **normal** or **anomaly (attack)** using three classification models, with an interactive Streamlit web interface.

**🔗 Live demo:** [https://ids-app.streamlit.app/](https://ids-app.streamlit.app/)

> The free hosting may put the app to sleep after a few days without visitors. If you see a "wake up" button, press it and wait about a minute while the models are trained again.

---

## Overview

| | |
|---|---|
| **Application** | Intrusion Detection |
| **Technique** | Classification (supervised learning), binary: `normal` / `anomaly` |
| **Dataset** | NSL-KDD (`KDDTrain+` and `KDDTest+`) |
| **Models** | Decision Tree, Random Forest, Gaussian Naive Bayes |
| **Interface** | Streamlit web app (also built as an ipywidgets form in Google Colab) |

## Features

- **Predict:** load a real record from the test set (random normal, random attack, or by row number) or type feature values by hand, then classify it. All three models are shown side by side, with the actual label when a real record is loaded.
- **Model comparison:** accuracy, precision, recall and F1-score, confusion matrices and classification reports.
- **Interface validation:** actual vs predicted for chosen test rows (default: 411, 0, 50, 100, 250, 500, 2000).
- **Dataset:** record counts, removed duplicates, class distribution and sample rows.

## Dataset

NSL-KDD is a refined version of the KDD Cup 1999 intrusion detection dataset. Each record is a network connection described by 41 features (basic connection, content, time-based and host-based traffic features) and a label.

| | Train (KDDTrain+) | Test (KDDTest+) |
|---|---|---|
| Raw records | 125,973 | 22,544 |
| After removing duplicates | 125,964 | 22,541 |
| Missing values | 0 | 0 |

The ARFF files are read with a small regex-based parser (`load_arff` in `ids_core.py`). Categorical features (`protocol_type`, `service`, `flag`) are label-encoded.

## Results

Evaluated on the held-out `KDDTest+` set (22,541 records):

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Decision Tree (max depth 10) | 77.22% | 83.47% | 77.22% | 76.92% |
| **Random Forest** (150 trees, max depth 15) | **77.97%** | 83.84% | 77.97% | 77.72% |
| Naive Bayes | 45.04% | 72.23% | 45.04% | 30.31% |

**Why is accuracy only around 77–78%?** `KDDTest+` deliberately contains attack types that never appear in the training data, so it measures real generalization rather than memorization. The models are very precise on the anomaly class (about 97%) but miss a share of new attacks (recall about 62–63%). Naive Bayes performs worst because it assumes the 41 features are independent, which is not true for network traffic.

## Project structure

```
ids-streamlit/
├── .streamlit/
│   └── config.toml        # theme (colours, fonts)
├── .gitignore
├── README.md
├── requirements.txt
├── app.py                 # Streamlit interface
├── ids_core.py            # data loading, preprocessing, training, prediction helpers
├── KDDTrain_.arff         # NSL-KDD training set
└── KDDTest_.arff          # NSL-KDD test set
```

## Run locally

```bash
# 1. clone the repository
git clone https://github.com/USERNAME/ids-streamlit.git
cd ids-streamlit

# 2. (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

# 3. install the dependencies
pip install -r requirements.txt

# 4. start the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

The first start trains the three models (about a minute) and saves them to `ids_model_cache.joblib`. Later starts load that file and open instantly. Delete the file if you change the data or the model settings.

## How it works

1. `ids_core.load_arff` reads the ARFF header and data into a pandas DataFrame.
2. Duplicates are removed, categorical features are label-encoded (fitted on train and test together to avoid unseen-category errors), and the target is encoded as `anomaly` / `normal`.
3. The three models are trained on `KDDTrain+` and evaluated on `KDDTest+`.
4. `app.py` builds the interface. A single connection is encoded in the same way as the training data and passed to every model.

## Notes and limitations

- Features you do not change in the form stay at `0` (or the first category), so loaded real records give more reliable predictions than hand-typed ones.
- The task is binary (`normal` / `anomaly`). The ARFF files do not contain the specific attack type (DoS, Probe, R2L, U2R).
- This is a learning project, not a production security tool.

## Tech stack

Python · pandas · NumPy · scikit-learn · matplotlib · Streamlit · joblib

## Acknowledgements

- NSL-KDD dataset: Tavallaee, Bagheri, Lu and Ghorbani, *A Detailed Analysis of the KDD CUP 99 Data Set* (2009), Canadian Institute for Cybersecurity, University of New Brunswick.
