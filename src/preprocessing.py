import numpy as np
import pandas as pd

def clean_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Cleans the input dataframe.
    - Removes outliers for training data.
    - Imputes missing values based on domain logic.
    """
    df = df.copy()
    
    # 1. Outlier removal (only for train data)
    if is_train:
        # Standard Ames outliers: GrLivArea > 4000 sq ft with low SalePrice
        df = df.drop(df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)].index)
        # Reset index after dropping
        df = df.reset_index(drop=True)
    
    # 2. Impute missing values
    
    # Categoricals where NA means "None" or "No Feature"
    none_cols = [
        'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'MasVnrType'
    ]
    for col in none_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')
            
    # Numericals where NA means 0 (mostly basement/garage features for homes without them)
    zero_cols = [
        'GarageYrBlt', 'GarageCars', 'GarageArea', 
        'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
        'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'
    ]
    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            
    # LotFrontage: Impute based on neighborhood median
    if 'LotFrontage' in df.columns and 'Neighborhood' in df.columns:
        df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
        # If any still missing (unlikely, but safe), impute with overall median
        df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
        
    # MSZoning: Impute with mode grouped by MSSubClass
    if 'MSZoning' in df.columns and 'MSSubClass' in df.columns:
        df['MSZoning'] = df.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'RL'))
    
    # Other categoricals/numericals with small missingness: impute with mode/median
    mode_cols = ['Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Utilities']
    for col in mode_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'TA')
            
    # Functional: Typical unless deductions are warranted
    if 'Functional' in df.columns:
        df['Functional'] = df['Functional'].fillna('Typ')
        
    return df
