# 💧 Intelligent Water Quality Assessment System

An end-to-end Machine Learning pipeline and interactive Streamlit web application designed to evaluate water safety and predict potability based on 9 physicochemical water quality parameters.

---

## 👥 Team & Collaboration
Developed as part of the Machine Learning course (ECE471) at Nile University:
* **Nasser Abdelfattah Alalem** (Coding, Modeling & Application)
* **Karim Osama Ibrahim** (Coding, Modeling & Application)
* **Mazen Osama Anwar** (Report & Evaluation)
* **Moaz Ahmed** (Research & Formulation)

---

## 📌 Project Overview
Access to clean drinking water is vital for human health. This project applies supervised machine learning to classify water samples as potable (Safe / Class 1) or non-potable (Unsafe / Class 0) using laboratory metrics:
* **pH**, **Hardness**, **Solids (TDS)**, **Chloramines**, **Sulfate**, **Conductivity**, **Organic Carbon**, **Trihalomethanes**, and **Turbidity**.

---

## ⚙️ Tech Stack & Methodology
* **Language & Frameworks:** Python, Scikit-Learn, Streamlit, Pandas, NumPy, Matplotlib, Seaborn.
* **Leakage-Free Preprocessing:** Embedded `SimpleImputer` (median strategy + missingness indicator flags) directly inside Scikit-Learn `Pipeline`s.
* **Class Imbalance Handling:** Managed the 61% to 39% class distribution using `class_weight='balanced'`.
* **Model Selection & Tuning:** Benchmarked Logistic Regression (49.54% accuracy) and RBF-Kernel SVM (61.28% accuracy) against an optimized **Random Forest Classifier** tuned via 5-Fold `GridSearchCV` (PR-AUC = 0.5893).
* **Feature Importance:** Permutation importance on the test set identified **Sulfate** and **pH** as the most critical predictors of potability.
* **Domain Guardrails:** Integrated rule-based chemical overrides in the web app to flag samples with caustic/extreme pH values (< 4.0 or > 10.0) as non-potable regardless of model output.

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone [https://github.com/NasserAlalem/water-quality-assessment-ml.git](https://github.com/NasserAlalem/water-quality-assessment-ml.git)
cd water-quality-assessment-ml
pip install -r requirements.txt
```

### 2. Run the Streamlit Web Application
```bash
streamlit run app.py
```

---

## 📁 Repository Structure
* `app.py`: Streamlit application code with manual slider controls and a pre-configured safe demo profile.
* `Water_Potability_Project.ipynb`: Jupyter notebook detailing exploratory data analysis, baselines, grid search, and evaluation.
* `data/water_potability.csv`: Cleaned dataset containing 3,276 water quality records.
* `docs/Final-Report.pdf`: Complete academic project report and architectural diagrams.
* `requirements.txt`: Python package requirements.