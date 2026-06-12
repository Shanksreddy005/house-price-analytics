import json

def main():
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 2 & 3: Data Cleaning & Feature Engineering\n",
                    "**Project**: House Price Analytics & Predictive Modeling\n",
                    "\n",
                    "This notebook documents and validates our data cleaning pipeline (outlier treatment, missing value imputation) and domain-driven feature engineering (16+ features)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import sys\n",
                    "sys.path.append('../src')\n",
                    "\n",
                    "from preprocessing import clean_data\n",
                    "from features import FeatureEngineer\n",
                    "\n",
                    "# 1. Load Raw Data\n",
                    "train = pd.read_csv('../data/train.csv')\n",
                    "print(f\"Raw Data Shape: {train.shape}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 1. Data Cleaning Pipeline\n",
                    "We run `clean_data` to treat outliers and impute missing values."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "train_cleaned = clean_data(train, is_train=True)\n",
                    "print(f\"Cleaned Data Shape: {train_cleaned.shape}\")\n",
                    "print(f\"Remaining Missing Values: {train_cleaned.isnull().sum().sum()}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2. Feature Engineering Pipeline\n",
                    "We instantiate the `FeatureEngineer` class, fit it to learn Neighborhood mappings and label encodings, and transform the cleaned dataset."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "engineer = FeatureEngineer()\n",
                    "engineer.fit(train_cleaned.drop(columns=['SalePrice']), target=train_cleaned['SalePrice'])\n",
                    "train_feat = engineer.transform(train_cleaned.drop(columns=['SalePrice']))\n",
                    "\n",
                    "print(f\"Engineered Data Shape: {train_feat.shape}\")\n",
                    "print(\"\\nEngineered Feature Set List:\")\n",
                    "print(train_feat.columns.tolist())\n",
                    "train_feat.head()"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open('notebooks/02_Feature_Engineering.ipynb', 'w') as f:
        json.dump(notebook_content, f, indent=2)
    print("Successfully created notebooks/02_Feature_Engineering.ipynb")

if __name__ == '__main__':
    main()
