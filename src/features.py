import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class FeatureEngineer:
    def __init__(self, new_features=None):
        self.neighborhood_map = {}
        self.label_encoders = {}
        self.categorical_cols = []
        self.new_features = new_features if new_features is not None else []
        
    def fit(self, df: pd.DataFrame, target: pd.Series = None):
        # 1. Learn Neighborhood price tiers based on median SalePrice
        if target is not None:
            temp_df = pd.DataFrame({'Neighborhood': df['Neighborhood'], 'SalePrice': target})
            medians = temp_df.groupby('Neighborhood')['SalePrice'].median().sort_values()
            self.neighborhood_map = {neigh: i for i, neigh in enumerate(medians.index)}
        else:
            # Fallback if target is not provided
            unique_neighs = df['Neighborhood'].unique()
            self.neighborhood_map = {neigh: i for i, neigh in enumerate(unique_neighs)}
            
        # Identify categorical columns
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Fit label encoders for remaining categoricals
        for col in self.categorical_cols:
            le = LabelEncoder()
            # Feed 'None' or empty placeholder to handle unseen classes
            le.fit(df[col].astype(str).tolist() + ['MissingClass'])
            self.label_encoders[col] = le
            
        return self
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # --- Engineer 15+ Domain Features ---
        
        # 1. TotalSF (Total square footage of house including basement)
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        
        # 2. HouseAge
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        df['HouseAge'] = df['HouseAge'].clip(lower=0) # Protect against weird data errors
        
        # 3. YearsSinceRemodel
        df['YearsSinceRemodel'] = df['YrSold'] - df['YearRemodAdd']
        df['YearsSinceRemodel'] = df['YearsSinceRemodel'].clip(lower=0)
        
        # 4. TotalBathrooms
        df['TotalBathrooms'] = (
            df['FullBath'] + 
            0.5 * df['HalfBath'] + 
            df['BsmtFullBath'] + 
            0.5 * df['BsmtHalfBath']
        )
        
        # 5. TotalPorchSF
        df['TotalPorchSF'] = (
            df['OpenPorchSF'] + 
            df['3SsnPorch'] + 
            df['EnclosedPorch'] + 
            df['ScreenPorch'] + 
            df['WoodDeckSF']
        )
        
        # 6. HasGarage
        df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
        
        # 7. HasBasement
        df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
        
        # 8. HasPool
        df['HasPool'] = (df['PoolArea'] > 0).astype(int)
        
        # 9. OverallQualityInteraction
        df['OverallQualityInteraction'] = df['OverallQual'] * df['OverallCond']
        
        # 10. NeighborhoodPriceTier
        df['NeighborhoodPriceTier'] = df['Neighborhood'].map(self.neighborhood_map).fillna(0).astype(int)
        
        # 11. OverallQualGrLivArea
        df['OverallQualGrLivArea'] = df['OverallQual'] * df['GrLivArea']
        
        # 12. YearBuiltRemod
        df['YearBuiltRemod'] = df['YearBuilt'] + df['YearRemodAdd']
        
        # 13. TotalRmsQual
        df['TotalRmsQual'] = df['TotRmsAbvGrd'] * df['OverallQual']
        
        # 14. IsNewHouse
        df['IsNewHouse'] = (df['YrSold'] == df['YearBuilt']).astype(int)
        
        # 15. GarageCarCapacity
        df['GarageCarCapacity'] = df['GarageCars'] * df['GarageArea']
        
        # 16. HighQualSF
        df['HighQualSF'] = df['1stFlrSF'] + df['2ndFlrSF']
        
        # --- Advanced Candidate Features for Ablation Studies ---
        if 'TotalHouseSF' in self.new_features:
            df['TotalHouseSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'] + df['GrLivArea']
            
        if 'QualityAgeInteraction' in self.new_features:
            df['QualityAgeInteraction'] = df['OverallQual'] * (1.0 / (df['HouseAge'] + 1.0))
            
        if 'FinishedBasementRatio' in self.new_features:
            df['FinishedBasementRatio'] = (df['BsmtFinSF1'] + df['BsmtFinSF2']) / (df['TotalBsmtSF'] + 1.0)
            
        if 'TotalBathPerRoom' in self.new_features:
            df['TotalBathPerRoom'] = df['TotalBathrooms'] / (df['TotRmsAbvGrd'] + 1e-5)
            
        if 'PorchToLotRatio' in self.new_features:
            df['PorchToLotRatio'] = df['TotalPorchSF'] / (df['LotArea'] + 1e-5)
            
        if 'GarageCarsPerRoom' in self.new_features:
            df['GarageCarsPerRoom'] = df['GarageCars'] / (df['TotRmsAbvGrd'] + 1e-5)
            
        if 'BasementRatio' in self.new_features:
            df['BasementRatio'] = df['TotalBsmtSF'] / (df['GrLivArea'] + 1e-5)
            
        if 'QualityPerArea' in self.new_features:
            df['QualityPerArea'] = df['OverallQual'] / (df['GrLivArea'] + 1e-5)
            
        if 'GarageRatio' in self.new_features:
            df['GarageRatio'] = df['GarageArea'] / (df['GrLivArea'] + 1e-5)
            
        if 'AgeBucket' in self.new_features:
            df['AgeBucket'] = pd.cut(df['HouseAge'], bins=[-1, 5, 20, 50, 999], labels=[0, 1, 2, 3]).astype(int)
        
        # Label Encode categorical features
        for col in self.categorical_cols:
            if col in df.columns:
                le = self.label_encoders[col]
                # Map unseen classes to 'MissingClass'
                df[col] = df[col].astype(str).map(lambda s: s if s in le.classes_ else 'MissingClass')
                df[col] = le.transform(df[col])
                
        # Drop columns that are completely uninformative (e.g. Id)
        if 'Id' in df.columns:
            df = df.drop(columns=['Id'])
            
        return df
