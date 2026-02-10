# ❤️ Heart Disease Classification — CNN Pipeline with Feature Synthesis

This repository contains the official implementation of our research work:

**“Convolutional Neural Network Pipeline with Feature Synthesis for Cardiovascular Risk Prediction”**

The included IEEE-format paper in this repository is the **official report** describing the methodology, experiments, and results of this project.

---

## 📄 Official Paper

📎 **File:**  
`IEEE_CAI___Convolutional_Neural_Network_Pipeline_with_Feature_Synthesis_for_Cardiovascular_Risk_Prediction.pdf`

This paper provides the complete technical details of:

- Dataset description  
- Preprocessing pipeline  
- Feature engineering & synthesis  
- ANOVA-based feature selection  
- 1D CNN architecture for tabular clinical data  
- Training strategy and evaluation metrics  
- Experimental results and analysis  

---

## 🧠 Project Overview

This project presents a **Convolutional Neural Network (CNN) pipeline for heart disease classification** using structured clinical (tabular) data. The approach combines:

- Outlier handling (IQR clipping)
- Clinically motivated feature engineering
- Interaction feature synthesis
- ANOVA F-value feature selection
- 1D CNN architecture tailored for tabular inputs

The goal is to build an accurate and computationally efficient model for **cardiovascular risk prediction**.

---

## ⚙️ Pipeline Summary

### 🔹 Data Preprocessing
- Outlier detection using IQR
- Outlier clipping instead of row removal
- Feature standardization (z-score normalization)
- Label encoding for categorical variables

### 🔹 Feature Engineering
Engineered features include:

- `age_group` — age binning
- `age_bp` — age × resting blood pressure
- `chol_age` — cholesterol / age ratio

### 🔹 Feature Selection
- ANOVA F-value ranking
- Top 15 features selected for training

### 🔹 Model
- 1D CNN for tabular data
- 3 Conv1D layers + BatchNorm + MaxPooling
- Fully connected layers with Dropout
- Adam optimizer + Early stopping + LR scheduler

---

## 📊 Dataset

- Source: Kaggle Heart Disease Dataset
- Samples: **1,025**
- Clinical attributes: **14**
- After engineering: **17 features**
- Selected for model: **15 features**
- Balanced class distribution (~51% positive)

---

## 🧪 Results (From Official Paper)

**Test Set Performance:**

| Metric | Value |
|--------|---------|
| Accuracy | **97.56%** |
| Precision | 97.17% |
| Recall (Sensitivity) | 98.10% |
| F1 Score | 97.63% |
| AUC–ROC | **99.80%** |
| Specificity | 97.00% |

- Test errors: **5 / 205**
- Training time: **~3.5 seconds**
- Train–test accuracy gap: **2.2%** (controlled overfitting)

---

## 📁 Repository Structure

