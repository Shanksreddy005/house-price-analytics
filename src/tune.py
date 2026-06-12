import os
import datetime
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from preprocessing import clean_data
from features import FeatureEngineer

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def evaluate_model(X_cleaned, y, train_cleaned, model_class, params, kf):
    oof = np.zeros(len(X_cleaned))
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cleaned, y)):
        X_tr_raw = X_cleaned.iloc[train_idx].copy()
        X_va_raw = X_cleaned.iloc[val_idx].copy()
        y_tr_raw = train_cleaned['SalePrice'].iloc[train_idx]
        
        engineer = FeatureEngineer()
        engineer.fit(X_tr_raw, target=y_tr_raw)
        X_tr = engineer.transform(X_tr_raw)
        X_va = engineer.transform(X_va_raw)
        
        y_tr = y.iloc[train_idx]
        
        fold_model = model_class(**params)
        fold_model.fit(X_tr, y_tr)
        oof[val_idx] = fold_model.predict(X_va)
        
    return rmse(y, oof)

def main():
    print("Starting Model Tuning Optimization...")
    train_path = 'data/train.csv'
    train_df = pd.read_csv(train_path)
    
    train_cleaned = clean_data(train_df, is_train=True)
    y = np.log1p(train_cleaned['SalePrice'])
    X_cleaned = train_cleaned.drop(columns=['SalePrice'])
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define Parameter Spaces
    cat_space = {
        'iterations': [1000, 1200, 1500],
        'learning_rate': [0.02, 0.03, 0.05],
        'depth': [4, 5, 6],
        'l2_leaf_reg': [2, 3, 5],
        'random_seed': [42],
        'verbose': [0]
    }
    
    xgb_space = {
        'n_estimators': [800, 1000, 1200],
        'learning_rate': [0.02, 0.03, 0.05],
        'max_depth': [3, 4, 5],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.3, 0.4, 0.5],
        'reg_alpha': [0.0, 0.1, 0.5],
        'reg_lambda': [1.0, 2.0],
        'random_state': [42],
        'verbosity': [0]
    }
    
    lgb_space = {
        'n_estimators': [800, 1000, 1200],
        'learning_rate': [0.02, 0.03, 0.05],
        'max_depth': [3, 4, 5],
        'num_leaves': [10, 15, 20],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.3, 0.4, 0.5],
        'random_state': [42],
        'verbose': [-1]
    }
    
    # We will sample 10 trials for each model to keep execution fast and efficient
    num_trials = 10
    
    best_models = {}
    
    # 1. Tune CatBoost
    print("\nTuning CatBoost...")
    best_cat_score = 999.0
    best_cat_params = None
    for trial in range(num_trials):
        params = {k: random.choice(v) for k, v in cat_space.items()}
        score = evaluate_model(X_cleaned, y, train_cleaned, CatBoostRegressor, params, kf)
        print(f"Trial {trial+1}/{num_trials}: {params} -> Score: {score:.6f}")
        if score < best_cat_score:
            best_cat_score = score
            best_cat_params = params
    print(f"Best CatBoost Score: {best_cat_score:.6f} with {best_cat_params}")
    best_models['CatBoost'] = (best_cat_score, best_cat_params)
    
    # 2. Tune XGBoost
    print("\nTuning XGBoost...")
    best_xgb_score = 999.0
    best_xgb_params = None
    for trial in range(num_trials):
        params = {k: random.choice(v) for k, v in xgb_space.items()}
        score = evaluate_model(X_cleaned, y, train_cleaned, XGBRegressor, params, kf)
        print(f"Trial {trial+1}/{num_trials}: {params} -> Score: {score:.6f}")
        if score < best_xgb_score:
            best_xgb_score = score
            best_xgb_params = params
    print(f"Best XGBoost Score: {best_xgb_score:.6f} with {best_xgb_params}")
    best_models['XGBoost'] = (best_xgb_score, best_xgb_params)
    
    # 3. Tune LightGBM
    print("\nTuning LightGBM...")
    best_lgb_score = 999.0
    best_lgb_params = None
    for trial in range(num_trials):
        params = {k: random.choice(v) for k, v in lgb_space.items()}
        score = evaluate_model(X_cleaned, y, train_cleaned, LGBMRegressor, params, kf)
        print(f"Trial {trial+1}/{num_trials}: {params} -> Score: {score:.6f}")
        if score < best_lgb_score:
            best_lgb_score = score
            best_lgb_params = params
    print(f"Best LightGBM Score: {best_lgb_score:.6f} with {best_lgb_params}")
    best_models['LightGBM'] = (best_lgb_score, best_lgb_params)
    
    # Save best parameters to a pickle file for training
    import pickle
    with open('outputs/models/best_tuned_params.pkl', 'wb') as f:
        pickle.dump(best_models, f)
    print("\nTuned parameters saved to outputs/models/best_tuned_params.pkl")

if __name__ == '__main__':
    main()
