# Tableau Dashboard Integration Guide

This directory contains resources for connecting the cleaned and engineered datasets to Tableau.

## File Contents
- [train_processed.csv](file:///c:/Users/ShaShank/OneDrive/Desktop/house-prices-advanced-regression-techniques/data/train_processed.csv): Cleaned and engineered training dataset including predictions and target variables.
- [test_processed.csv](file:///c:/Users/ShaShank/OneDrive/Desktop/house-prices-advanced-regression-techniques/data/test_processed.csv): Cleaned and engineered testing dataset.

## How to Set Up the Dashboard in Tableau
1. Open **Tableau Desktop** (or Tableau Public).
2. Connect to the text file `data/train_processed.csv`.
3. Replicate the design schema using the following sheet parameters:

### 1. Key Performance Indicators (KPIs)
Create calculated fields:
- **Average Sale Price**: `AVG([SalePrice])` (formatted as Currency, US Dollars)
- **Median Sale Price**: `MEDIAN([SalePrice])` (formatted as Currency, US Dollars)
- **Average House Age**: `AVG([HouseAge])`
- **Average Living Area**: `AVG([GrLivArea])` (formatted in sq ft)

### 2. Neighborhood vs. Sale Price Chart
- **Columns**: `Neighborhood` (sorted descending by average SalePrice)
- **Rows**: `SalePrice` (AVG)
- **Mark**: Bar Chart (color-graded by Average SalePrice to show price tier clusters)

### 3. Overall Quality vs. Sale Price
- **Columns**: `OverallQual` (treated as a dimension)
- **Rows**: `SalePrice`
- **Mark**: Box-and-Whisker plot to illustrate price range variances.

### 4. House Age Distribution
- **Columns**: `HouseAge` (binned in 5 or 10-year intervals)
- **Rows**: `CNT(Id)` or `CNT(train_processed.csv)`
- **Mark**: Area chart or Histogram.

### 5. Feature Importance Visual
- Plot the SHAP global importances from the `outputs/feature_importance/` directory to show recruiters the feature ranks.
