import os
import datetime
import numpy as np
import pandas as pd
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

def evaluate_pipeline(train_df, new_features=None, verbose=True):
    """
    Evaluates the feature set using a leak-free 5-Fold Cross Validation.
    Fits and transforms the FeatureEngineer strictly inside each training fold.
    """
    train_cleaned = clean_data(train_df, is_train=True)
    y = np.log1p(train_cleaned['SalePrice'])
    X_cleaned = train_cleaned.drop(columns=['SalePrice'])
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # We will track OOF predictions for the leading model (CatBoost) to test feature value
    model = CatBoostRegressor(iterations=1200, learning_rate=0.03, depth=5, random_seed=42, verbose=0)
    
    oof = np.zeros(len(X_cleaned))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_cleaned, y)):
        X_tr_raw = X_cleaned.iloc[train_idx].copy()
        X_va_raw = X_cleaned.iloc[val_idx].copy()
        y_tr_raw = train_cleaned['SalePrice'].iloc[train_idx]
        
        # Fit inside fold
        engineer = FeatureEngineer(new_features=new_features)
        engineer.fit(X_tr_raw, target=y_tr_raw)
        
        X_tr = engineer.transform(X_tr_raw)
        X_va = engineer.transform(X_va_raw)
        
        y_tr = y.iloc[train_idx]
        
        # Fit model
        fold_model = CatBoostRegressor(iterations=1200, learning_rate=0.03, depth=5, random_seed=42, verbose=0)
        fold_model.fit(X_tr, y_tr)
        
        oof[val_idx] = fold_model.predict(X_va)
        
    score = rmse(y, oof)
    if verbose:
        print(f"Features: {new_features} -> CatBoost OOF RMSLE: {score:.6f}")
    return score

def main():
    print("Starting Feature Ablation Study (Leakage-Free CV Baseline)...")
    train_path = 'data/train.csv'
    if not os.path.exists(train_path):
        raise FileNotFoundError("Missing train.csv")
    train_df = pd.read_csv(train_path)
    
    # 1. Establish Baseline (0 engineered features from optimization tiers)
    baseline_score = evaluate_pipeline(train_df, new_features=[], verbose=True)
    
    # Tiers of candidate features to evaluate
    tier_1 = ['TotalHouseSF', 'QualityAgeInteraction', 'FinishedBasementRatio', 'TotalBathPerRoom']
    tier_2 = ['PorchToLotRatio', 'GarageCarsPerRoom', 'BasementRatio']
    tier_3 = ['QualityPerArea', 'GarageRatio', 'AgeBucket']
    
    selected_features = []
    current_best_score = baseline_score
    
    # Create experiments row function
    def log_experiment(feat_list, score, notes):
        exp_file = 'outputs/experiments.csv'
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_row = pd.DataFrame([{
            'Timestamp': timestamp,
            'Model': 'CatBoost_Ablation',
            'CV_RMSLE': round(score, 6),
            'FeatureSet_Version': f"Ablation: {feat_list}",
            'Hyperparameters': 'iterations=1200, lr=0.03, depth=5',
            'Notes': notes
        }])
        if os.path.exists(exp_file):
            old_df = pd.read_csv(exp_file)
            new_df = pd.concat([old_df, new_row], ignore_index=True)
        else:
            new_df = new_row
        new_df.to_csv(exp_file, index=False)
        
    log_experiment([], baseline_score, "Leak-free CV Baseline")
    
    # Process Tiers sequentially
    for tier_num, tier_feats in [(1, tier_1), (2, tier_2), (3, tier_3)]:
        print(f"\nEvaluating Tier {tier_num} features: {tier_feats}")
        any_improvement = False
        
        for feat in tier_feats:
            candidate_feats = selected_features + [feat]
            score = evaluate_pipeline(train_df, new_features=candidate_feats, verbose=True)
            
            # If the score improves by at least a tiny fraction
            if score < current_best_score:
                diff = current_best_score - score
                print(f"  [RETAINED] {feat} (Improved CV by {diff:.6f})")
                selected_features.append(feat)
                current_best_score = score
                any_improvement = True
                log_experiment(selected_features, score, f"Added {feat} (Tier {tier_num})")
            else:
                print(f"  [REJECTED] {feat} (Worsened or no change)")
                
        # If no features in this tier improved the score, stop proceeding to the next tier
        if not any_improvement:
            print(f"No improvements found in Tier {tier_num}. Stopping feature evaluation.")
            break
            
    print(f"\nFinal Selected Features: {selected_features}")
    print(f"Final Optimized Feature Set CV Score: {current_best_score:.6f}")

if __name__ == '__main__':
    main()
