import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Starting EDA Visualizations and Notebook Generation...")
    
    # Ensure directories exist
    os.makedirs('outputs/visualizations', exist_ok=True)
    os.makedirs('notebooks', exist_ok=True)
    
    # Load raw data
    train_path = 'data/train.csv'
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing train data at {train_path}")
        
    df = pd.read_csv(train_path)
    
    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    
    # 1. Price Distribution (Original vs Log Transformed)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.histplot(df['SalePrice'], kde=True, ax=axes[0], color='royalblue')
    axes[0].set_title('Original SalePrice Distribution (Skewed)', fontsize=14)
    axes[0].set_xlabel('SalePrice ($)', fontsize=12)
    
    sns.histplot(np.log1p(df['SalePrice']), kde=True, ax=axes[1], color='forestgreen')
    axes[1].set_title('Log-Transformed SalePrice (Normal)', fontsize=14)
    axes[1].set_xlabel('Log(SalePrice + 1)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('outputs/visualizations/price_distribution.png', dpi=300)
    plt.close()
    print("Generated price_distribution.png")
    
    # 2. Correlation Heatmap (Top 15 features)
    # Select only numeric features
    numeric_df = df.select_dtypes(include=[np.number])
    top_corr_features = numeric_df.corr()['SalePrice'].sort_values(ascending=False).head(15).index
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df[top_corr_features].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Top 15 Correlated Features with SalePrice', fontsize=16)
    plt.tight_layout()
    plt.savefig('outputs/visualizations/correlation_heatmap.png', dpi=300)
    plt.close()
    print("Generated correlation_heatmap.png")
    
    # 3. Neighborhood vs Price Boxplot
    plt.figure(figsize=(16, 8))
    # Sort neighborhoods by median sale price
    neigh_order = df.groupby('Neighborhood')['SalePrice'].median().sort_values().index
    sns.boxplot(x='Neighborhood', y='SalePrice', data=df, order=neigh_order, palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.title('Sale Price Distribution across Neighborhoods (Sorted by Median)', fontsize=16)
    plt.xlabel('Neighborhood', fontsize=12)
    plt.ylabel('SalePrice ($)', fontsize=12)
    plt.tight_layout()
    plt.savefig('outputs/visualizations/neighborhood_vs_price.png', dpi=300)
    plt.close()
    print("Generated neighborhood_vs_price.png")
    
    # 4. Quality vs Price Boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='OverallQual', y='SalePrice', data=df, palette='magma')
    plt.title('Overall Quality vs SalePrice', fontsize=16)
    plt.xlabel('Overall Quality (1-10)', fontsize=12)
    plt.ylabel('SalePrice ($)', fontsize=12)
    plt.tight_layout()
    plt.savefig('outputs/visualizations/quality_vs_price.png', dpi=300)
    plt.close()
    print("Generated quality_vs_price.png")
    
    # 5. Missing Values Analysis
    missing_series = df.isnull().sum()
    missing_series = missing_series[missing_series > 0].sort_values(ascending=False)
    missing_percent = (missing_series / len(df)) * 100
    
    if not missing_series.empty:
        plt.figure(figsize=(14, 6))
        sns.barplot(x=missing_percent.index, y=missing_percent.values, palette='coolwarm')
        plt.xticks(rotation=90)
        plt.title('Percentage of Missing Values per Feature', fontsize=16)
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('% Missing', fontsize=12)
        plt.tight_layout()
        plt.savefig('outputs/visualizations/missing_values_analysis.png', dpi=300)
        plt.close()
        print("Generated missing_values_analysis.png")
    else:
        print("No missing values found to plot.")
        
    # --- Generate 01_EDA.ipynb Notebook ---
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 1: Exploratory Data Analysis (EDA)\n",
                    "**Project**: House Price Analytics & Predictive Modeling\n",
                    "\n",
                    "This notebook provides a thorough exploratory analysis of the Ames housing dataset to address key recruiter-facing business questions and discover insights for predictive modeling."
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
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "%matplotlib inline\n",
                    "sns.set_theme(style='whitegrid')\n",
                    "\n",
                    "# Load the dataset\n",
                    "train = pd.read_csv('../data/train.csv')\n",
                    "print(f'Train shape: {train.shape}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 1. Target Variable Analysis: SalePrice\n",
                    "Let's examine the distribution of `SalePrice` and look at skewness. Evaluating target transformation is a key step."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "print(f\"Original SalePrice Skewness: {train['SalePrice'].skew():.4f}\")\n",
                    "print(f\"Log-Transformed SalePrice Skewness: {np.log1p(train['SalePrice']).skew():.4f}\")\n",
                    "\n",
                    "# Load and show pre-generated image\n",
                    "from IPython.display import Image, display\n",
                    "display(Image(filename='../outputs/visualizations/price_distribution.png'))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 2. Feature Correlations\n",
                    "Let's check which variables correlate strongest with house prices."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "display(Image(filename='../outputs/visualizations/correlation_heatmap.png'))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 3. Neighborhood Dynamics\n",
                    "How do neighborhoods influence pricing? We plot the distributions sorted by median."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "display(Image(filename='../outputs/visualizations/neighborhood_vs_price.png'))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 4. Overall Quality vs SalePrice\n",
                    "Let's check how strongly material and finish rating correlates with pricing."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "display(Image(filename='../outputs/visualizations/quality_vs_price.png'))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### 5. Missing Values Analysis\n",
                    "Analyzing which columns have missing values and their counts."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "display(Image(filename='../outputs/visualizations/missing_values_analysis.png'))"
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
    
    with open('notebooks/01_EDA.ipynb', 'w') as f:
        json.dump(notebook_content, f, indent=2)
    print("Successfully created notebooks/01_EDA.ipynb")

if __name__ == '__main__':
    main()
