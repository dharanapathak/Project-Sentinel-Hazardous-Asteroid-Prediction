# Project Sentinel — Hazardous Asteroid Prediction

An educational machine-learning project that predicts the dataset-defined `hazardous` / `non-hazardous` label for Near-Earth Object (NEO) records using tabular characteristics.

> **Scientific limitation:** This is a supervised-learning exercise on historical dataset labels. It is not an official NASA/JPL risk assessment, orbital calculation, impact-probability model, or planetary-defence tool. Do not use its predictions for real-world safety decisions.

## Features
- Data validation and preprocessing
- Exploratory data analysis in the original Jupyter notebook
- Stratified train/test split
- Logistic Regression and Random Forest model comparison
- Class-imbalance handling with class weights
- Cross-validation and compact Random Forest hyperparameter search
- Held-out evaluation: precision, recall, F1, ROC-AUC, average precision and confusion matrix
- Permutation feature importance
- Reproducible training script that saves the fitted preprocessing/model pipeline
- Streamlit interface to upload a CSV, train a model, inspect metrics and try predictions
- Unit tests for data preparation and inference input validation

## Repository layout
```text
Project-Sentinel-Hazardous-Asteroid-Prediction/
├── app.py
├── train.py
├── requirements.txt
├── .gitignore
├── README.md
├── data/
│   └── README.md
├── models/
│   └── .gitkeep
├── notebooks/
│   └── hazardous_asteroid_prediction_final.ipynb
├── src/
│   ├── __init__.py
│   ├── modeling.py
│   └── preprocessing.py
└── tests/
    ├── test_preprocessing.py
    └── test_modeling.py
```

## Dataset
The notebook you supplied was executed against a CSV with **90,836 rows and 10 columns**, including the target column `hazardous`. The CSV itself was not included with the uploaded notebook, so it is intentionally not fabricated or bundled here.

Place the authorized dataset file at `data/neo.csv`. The CSV should include a binary/boolean `hazardous` column and fields such as:
- `est_diameter_min`
- `est_diameter_max`
- `relative_velocity`
- `miss_distance`
- `absolute_magnitude`

The training pipeline excludes identifiers and descriptive fields such as `id`, `name`, `orbiting_body`, and `sentry_object` from model inputs. See `data/README.md` for dataset handling guidance. Only upload the dataset to GitHub if its license and source terms permit redistribution.

## Setup (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the model
After placing the CSV at `data/neo.csv`:
```powershell
python train.py --data data/neo.csv --output models/asteroid_pipeline.joblib
```
The script prints held-out metrics and saves a single fitted pipeline containing preprocessing and the selected estimator.

## Run the interactive app
```powershell
streamlit run app.py
```
Upload the NEO CSV in the app, train the model, then enter the predictor values to get a model output. The app also supports loading a previously trained `models/asteroid_pipeline.joblib` file if present.

## Run tests
```powershell
pytest -q
```

## Results recorded in the supplied notebook
The notebook's saved output reports the following held-out test metrics from its previous execution:
| Metric | Recorded value |
|---|---:|
| Accuracy | 0.785942 |
| Precision (hazardous) | 0.304227 |
| Recall (hazardous) | 0.932127 |
| F1 (hazardous) | 0.458733 |
| ROC-AUC | 0.879441 |
| Average Precision | 0.326208 |

These are historical saved notebook outputs, not a guarantee that a fresh run will reproduce identical values. Results can vary if the dataset or software versions differ. The relatively low hazardous-class precision means many positive predictions are false positives; accuracy alone is not sufficient for judging this imbalanced classification task.

## Important notes
- The saved `.joblib` model artifact was not present in the uploaded files. Generate it by running `train.py`.
- The source dataset was not present in the upload. Obtain it from the source authorized for your assignment and keep it in `data/neo.csv`.
- Do not commit private data, API keys, virtual environments, or large files without checking repository and dataset permissions.
- Model probabilities are classifier scores for the dataset label, not calibrated probabilities of an asteroid impact.
