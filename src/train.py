import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from preprocessing import clean_data
from features import FeatureEngineer

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    print("Starting Training Pipeline...")
    
    # Ensure directories exist
    os.makedirs('outputs/submissions', exist_ok=True)
    os.makedirs('outputs/feature_importance', exist_ok=True)
    os.makedirs('outputs/visualizations', exist_ok=True)
    os.makedirs('outputs/models', exist_ok=True)
    
    # 1. Load Data
    train_path = 'data/train.csv'
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing train data at {train_path}")
        
    train_df = pd.read_csv(train_path)
    print(f"Loaded train data. Shape: {train_df.shape}")
    
    # 2. Preprocessing
    train_cleaned = clean_data(train_df, is_train=True)
    y = np.log1p(train_cleaned['SalePrice'])
    X_cleaned = train_cleaned.drop(columns=['SalePrice'])
    
    # 3. Feature Engineering - Fit global for reference / Tableau
    engineer = FeatureEngineer()
    engineer.fit(X_cleaned, target=train_cleaned['SalePrice'])
    X_feat_global = engineer.transform(X_cleaned)
    print(f"Global features ready. Shape: {X_feat_global.shape}")
    
    # Save the global feature engineer
    with open('outputs/models/feature_engineer.pkl', 'wb') as f:
        pickle.dump(engineer, f)
        
    # Save processed features to data folder for easy access
    X_feat_with_target = X_feat_global.copy()
    X_feat_with_target['SalePrice_log'] = y
    X_feat_with_target['SalePrice'] = train_cleaned['SalePrice']
    X_feat_with_target.to_csv('data/train_processed.csv', index=False)
    
    # Load tuned parameters if available
    tuned_params_path = 'outputs/models/best_tuned_params.pkl'
    if os.path.exists(tuned_params_path):
        with open(tuned_params_path, 'rb') as f:
            tuned_params = pickle.load(f)
            print("Loaded tuned hyperparameters:", {k: v[1] for k, v in tuned_params.items()})
    else:
        tuned_params = {}
        
    # Define models to train
    models_dict = {
        'Ridge': Ridge(alpha=10.0),
        'Lasso': Lasso(alpha=0.0005, max_iter=10000),
        'ElasticNet': ElasticNet(alpha=0.0005, l1_ratio=0.5, max_iter=10000),
        'RandomForest': RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(**tuned_params['XGBoost'][1]) if 'XGBoost' in tuned_params else XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.4, random_state=42, verbosity=0),
        'LightGBM': LGBMRegressor(**tuned_params['LightGBM'][1]) if 'LightGBM' in tuned_params else LGBMRegressor(n_estimators=1000, learning_rate=0.03, max_depth=4, num_leaves=15, subsample=0.8, colsample_bytree=0.4, random_state=42, verbose=-1),
        'CatBoost': CatBoostRegressor(**tuned_params['CatBoost'][1]) if 'CatBoost' in tuned_params else CatBoostRegressor(iterations=1200, learning_rate=0.03, depth=5, random_seed=42, verbose=0)
    }
    
    # 4. Prepare Fold-Specific Datasets (Leakage-Free)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_datasets = []
    
    print("\nPreparing fold-specific feature sets (Leakage-Free)...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cleaned, y)):
        X_tr_raw = X_cleaned.iloc[train_idx].copy()
        X_va_raw = X_cleaned.iloc[val_idx].copy()
        y_tr_raw = train_cleaned['SalePrice'].iloc[train_idx]
        
        # Fit inside fold
        fold_engineer = FeatureEngineer()
        fold_engineer.fit(X_tr_raw, target=y_tr_raw)
        
        X_tr_fold = fold_engineer.transform(X_tr_raw)
        X_va_fold = fold_engineer.transform(X_va_raw)
        
        # Save fold feature engineer
        fold_eng_path = f'outputs/models/feature_engineer_fold_{fold}.pkl'
        with open(fold_eng_path, 'wb') as f:
            pickle.dump(fold_engineer, f)
            
        fold_datasets.append((X_tr_fold, X_va_fold, train_idx, val_idx))
    
    oof_predictions = {}
    cv_scores = {}
    
    # Run K-Fold training for each model
    for model_name, model in models_dict.items():
        print(f"\nTraining {model_name}...")
        oof = np.zeros(len(X_cleaned))
        fold_models = []
        
        for fold, (X_tr_fold, X_va_fold, train_idx, val_idx) in enumerate(fold_datasets):
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create a clone/copy of the estimator
            from sklearn.base import clone
            fold_model = clone(model)
            
            # Fit model
            fold_model.fit(X_tr_fold, y_tr)
            
            # Predict validation
            preds = fold_model.predict(X_va_fold)
            oof[val_idx] = preds
            
            fold_models.append(fold_model)
            
        score = rmse(y, oof)
        print(f"{model_name} OOF RMSLE: {score:.5f}")
        
        oof_predictions[model_name] = oof
        cv_scores[model_name] = score
        
        # Save fold models
        for fold, fold_model in enumerate(fold_models):
            with open(f'outputs/models/{model_name}_fold_{fold}.pkl', 'wb') as f:
                pickle.dump(fold_model, f)
                
    # Create experiments.csv or append to it
    exp_file = 'outputs/experiments.csv'
    exp_exists = os.path.exists(exp_file)
    
    exp_rows = []
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for model_name, score in cv_scores.items():
        exp_rows.append({
            'Timestamp': timestamp,
            'Model': model_name,
            'CV_RMSLE': round(score, 5),
            'FeatureSet_Version': 'v1.0 (16 Engineered Features - Leak Free CV)',
            'Hyperparameters': str(models_dict[model_name].get_params()),
            'Notes': 'Leak-free CV optimized run'
        })
        
    exp_df = pd.DataFrame(exp_rows)
    if exp_exists:
        old_df = pd.read_csv(exp_file)
        exp_df = pd.concat([old_df, exp_df], ignore_index=True)
    exp_df.to_csv(exp_file, index=False)
    print(f"\nSaved experiments to {exp_file}")
    
    # Save OOF predictions to a file for ensemble training
    oof_df = pd.DataFrame(oof_predictions)
    oof_df['Target'] = y
    oof_df.to_csv('outputs/oof_predictions.csv', index=False)
    print("Saved Out-Of-Fold predictions for ensembling.")
    
if __name__ == '__main__':
    main()
