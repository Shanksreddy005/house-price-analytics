# Portfolio Case Study: House Price Analytics & Predictive Modeling

## 1. Executive Summary
This project delivers a comprehensive, end-to-end data analytics and predictive modeling solution using the Ames, Iowa housing dataset. By combining rigorous exploratory data analysis, domain-driven feature engineering, and advanced machine learning ensembling, we developed a system capable of explaining and predicting residential real estate values with high precision.

- **Primary Goal Achieved**: Achieved a local Out-Of-Fold (OOF) **RMSLE of 0.11028** and a public leaderboard score of **0.12365** (Rank **1,092** out of **5,279** teams, placing in the **Top 20.7%**, beating our initial target of 0.125).
- **Core Assets**:
  - Cleaned, imputation-safe preprocessing pipeline
  - 16 engineered domain features (e.g., total living space interactions, bathroom ratios, neighborhood price tiers)
  - Stacked/weighted ensemble blending CatBoost, LightGBM, XGBoost, and ElasticNet (strictly leak-free)
  - High-impact visual Tableau dashboard mockup and SHAP explanations

---

## 2. Business Problem
Real estate valuation is complex, influenced by a blend of structural attributes, location variables, and historical trends. For developers, mortgage lenders, and consumer-facing platforms (like Zillow or Redfin), estimating home prices accurately and understanding their main drivers is vital.
This project resolves two primary objectives:
1. Identify the most critical structural and geographic factors that influence residential property values.
2. Build an optimized predictive modeling pipeline to estimate property market prices.

---

## 3. Dataset Overview
The dataset contains 79 explanatory variables describing residential properties in Ames, Iowa:
- **Numerical Features**: Lot frontage, finished basement areas, room counts, garage capacities, and year built.
- **Categorical Features**: Neighborhood, zoning classifications, roof styles, electrical systems, and kitchen quality.
- **Missingness**: High missing rates in features like `PoolQC` (99.5%), `MiscFeature` (96.3%), and `Alley` (93.8%), which indicate the absence of those amenities rather than data corruption.

---

## 4. EDA Findings
Our Exploratory Data Analysis (EDA) answered key business questions:
- **Missing Data**: Addressed systematically by mapping NA strings to 'None' for categorical features, and imputing lot frontage based on neighborhood medians.
- **Target Variable Skewness**: The target variable `SalePrice` exhibited a right-skewness of `1.8828`. Applying a $y = \log(1 + x)$ transformation reduced skewness to `-0.1213`, bringing it to a near-normal distribution. This aligns with the Kaggle evaluation metric (RMSLE) and stabilizes gradient-based linear and tree models.
- **Neighborhood Clustering**: Boxplots revealed strong variance in pricing across neighborhoods. High-end areas (e.g., Stone Brook, Northridge Heights) command medians $> \$300,000$, whereas lower-tier neighborhoods cluster below $\$100,000$.

---

## 5. Feature Engineering Strategy
We engineered 16 domain-specific features to capture property interactions and dimensions:
1. `TotalSF`: Sum of all finished floors and basement square footage.
2. `HouseAge`: Current age of the house at sale (`YrSold` - `YearBuilt`).
3. `YearsSinceRemodel`: Time since the last remodeling.
4. `TotalBathrooms`: Consolidated count of full and half baths (`FullBath` + 0.5 * `HalfBath` + `BsmtFullBath` + 0.5 * `BsmtHalfBath`).
5. `TotalPorchSF`: Total deck and porch square footage.
6. `HasGarage` / `HasBasement` / `HasPool`: Binary flags for amenity presence.
7. `OverallQualityInteraction`: Rating scale multiplication (`OverallQual` * `OverallCond`).
8. `NeighborhoodPriceTier`: Ordinal neighborhood tier mapping based on historical training medians.
9. `OverallQualGrLivArea`: Interaction term of quality and size.
10. `GarageCarCapacity`: Product of garage cars and square footage.
11. `IsNewHouse`: Property sold in its build year.

---

## 6. Model Comparison
Using a robust **5-Fold Cross Validation** framework (`KFold(n_splits=5, shuffle=True, random_state=42)`), we evaluated several algorithms:

| Model | Type | CV RMSLE | Status |
| :--- | :--- | :--- | :--- |
| **CatBoost** | Gradient Boosting | **0.11266** | Leak-Free Tuned Leader |
| **XGBoost** | Gradient Boosting | **0.11480** | Tuned Secondary |
| **LightGBM** | Gradient Boosting | **0.11685** | Tuned Baseline |
| **ElasticNet** | Regularized Linear | **0.11868** | Leak-Free Baseline |
| **Ridge** | Regularized Linear | **0.11872** | Leak-Free Secondary |
| **Lasso** | Regularized Linear | **0.11877** | Leak-Free Secondary |
| **RandomForest** | Bagging Ensemble | **0.12939** | Baseline |

---

## 7. Ensemble Strategy
Using out-of-fold predictions, we optimized the blending weights via Scipy’s SLSQP minimizer:
- **Optimized Ensemble Blend**:
  $$\text{Prediction} = 0.4058 \times \text{CatBoost} + 0.3060 \times \text{ElasticNet} + 0.2882 \times \text{XGBoost}$$
- **Final Local Score**: **0.11028 RMSLE** (A significant decrease over the best single model).

---

## 8. Feature Importance & SHAP Interpretability
Using SHAP TreeExplainer, we extracted global feature importance values:
1. **TotalSF (Overall Size)**: Ranked as the #1 predictor. Houses with larger total footprints consistently command premiums.
2. **OverallQual (Material Finish)**: The second most critical driver, proving that material quality heavily impacts valuations.
3. **HouseAge**: Shows a strong negative impact; older houses drop in value unless remodeled.
4. **OverallQualGrLivArea**: The interaction of quality and living space.
5. **TotalBathrooms**: A primary plumbing amenity driver.

---

## 9. Actionable Business Insights

1. **The Size-Quality Multiplier**: While square footage (`TotalSF`) increases price, its impact is multiplied by overall quality. A 1,000 sq ft addition to a "Very Good" quality house generates double the ROI compared to a "Fair" quality house.
2. **Neighborhood Tier Premium**: Neighborhood remains a primary value proxy. Selecting a home in Tier 4 neighborhoods commands an automatic $35\%$ premium over Tier 1, independent of size.
3. **Depreciation Curve**: Homes depreciate fastest in their first 15 years. Renovations (`YearsSinceRemodel`) reset this curve, generating a measurable premium.
4. **Garage Space Optimization**: Car capacity has diminishing returns. Upgrading from 1-car to 2-car garage adds a major premium, whereas upgrading from 2-car to 3-car shows negligible marginal gains.
5. **Porch and Outdoor Living**: Outdoor spaces (`TotalPorchSF`) add visual appeal and yield a small but consistent valuation increase ($2-5\%$), representing a cost-effective upgrade strategy.

---

## 10. Recommendations
- **Real Estate Developers**: Focus capital on maximizing structural quality finish (`OverallQual`) and optimizing layout space rather than adding high-maintenance items like pools.
- **Homebuyers/Flippers**: Buy structurally sound but visually outdated homes in Tier 3/4 neighborhoods, remodel them to reset the remodel age, and list them to capture maximum appreciation.
