# Flu Shot Prediction

This repository contains code for predicting vaccination status for H1N1 and Seasonal flu vaccines using machine learning models, including Random Forest, XGBoost, Logistic Regression, and LightGBM.

## Project Overview

The goal is to predict whether respondents are likely to receive H1N1 and Seasonal flu vaccines based on survey features. The models leverage feature preprocessing pipelines and multiple classifiers for robust predictions.

## Repository Structure

- `src/` - Python scripts for data preprocessing, model training, evaluation, and prediction.
- `notebooks/` - Jupyter notebooks for data analysis and visualization.
- `requirements.txt` - Python package dependencies.
- `main.py` - Entry script to run the full pipeline (preprocessing, training, evaluation, prediction).

## Data

Due to confidentiality, this repository **does not** include the original dataset. 

To run the code, please prepare your own datasets with the following expected files:

- `training_set_features.csv` — training features with a `respondent_id` column.
- `training_set_labels.csv` — training labels with `respondent_id` and target columns `h1n1_vaccine` and `seasonal_vaccine`.
- `test_set_features.csv` — test features with a `respondent_id` column.

**The feature columns and data format should match those described in the code for proper functioning.**

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/Coach-17-Ron/Kgabe_Flu-Shot-Prediction.git
   cd Kgabe_Flu-Shot-Prediction
