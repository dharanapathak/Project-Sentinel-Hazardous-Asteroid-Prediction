# 🚀 Project Sentinel – Hazardous Asteroid Prediction

An end-to-end Machine Learning project that predicts potentially hazardous asteroids using Near-Earth Object (NEO) data. The project applies data preprocessing, exploratory data analysis, feature engineering, and classification algorithms to identify patterns associated with potentially hazardous asteroids.

## 📌 Project Overview

Project Sentinel aims to build a Machine Learning classification system that analyzes asteroid-related features and predicts whether an asteroid is potentially hazardous or non-hazardous.

The project demonstrates how Machine Learning techniques can be applied to astronomical datasets for data analysis and classification.

## 🎯 Objectives

- Analyze Near-Earth Object (NEO) data.
- Perform data cleaning and preprocessing.
- Explore important patterns through EDA.
- Build classification models.
- Evaluate model performance using standard metrics.
- Develop an interactive prediction interface.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Algorithms:** Logistic Regression, Random Forest
- **Application:** Streamlit
- **Model Evaluation:** Accuracy, Precision, Recall, F1-score, ROC-AUC

## 📂 Project Structure

```text
Project-Sentinel-Hazardous-Asteroid-Prediction/
│
├── data/
│   └── README.md
│
├── models/
│   └── .gitkeep
│
├── notebooks/
│   └── hazardous_asteroid_prediction_final.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── modeling.py
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_modeling.py
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Project-Sentinel-Hazardous-Asteroid-Prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd Project-Sentinel-Hazardous-Asteroid-Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 📊 Dataset Setup

Place the authorized Near-Earth Object dataset CSV file inside the `data/` directory.

Expected file path:

```text
data/neo.csv
```

The dataset is not included in this repository. Make sure the dataset has the columns expected by the preprocessing and training scripts.

## 🚀 Run the Project

### Train the Machine Learning Models

```bash
python train.py
```

This runs the training and evaluation pipeline and saves the trained model when the required dataset is available.

### Launch the Streamlit Application

```bash
streamlit run app.py
```

## 🤖 Machine Learning Models

The project explores the following classification algorithms:

1. **Logistic Regression:** A baseline classification algorithm.
2. **Random Forest Classifier:** An ensemble learning algorithm used for classification.

Hyperparameter tuning and cross-validation are included to improve model evaluation.

## 📈 Model Evaluation

The notebook evaluates the models using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Feature Importance

The notebook's reported evaluation metrics include:

| Metric | Result |
|---|---:|
| Accuracy | 78.59% |
| Hazardous Class Precision | 30.42% |
| Hazardous Class Recall | 93.21% |
| Hazardous Class F1-score | 45.87% |
| ROC-AUC | 87.94% |

**Note:** These are results from the notebook's evaluation. The model is an educational classification project and must not be used for real-world asteroid impact-risk decisions.

## 🧪 Run Tests

```bash
pytest -q
```

## 🔮 Future Enhancements

- Integrate updated astronomical datasets.
- Experiment with advanced Machine Learning algorithms.
- Improve hazardous-class precision.
- Add more interactive data visualizations.
- Deploy the application on a cloud platform.
- Explore deep learning approaches for classification.

## 👩‍💻 Author

**Dharana Pathak**

Machine Learning | Python | Artificial Intelligence | Data Science

## ⭐ Acknowledgements

This project was developed as a hands-on Machine Learning project to explore data preprocessing, classification, model evaluation, and interactive application development.

---

If you find this project interesting, feel free to explore the repository and share your feedback!

**⭐ Don't forget to star the repository!**
