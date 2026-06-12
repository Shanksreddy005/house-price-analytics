import os
import numpy as np
import pandas as pd
import pickle
from preprocessing import clean_data

def predict_blend(test_cleaned, models_to_use, weights):
    ensemble_preds = np.zeros(len(test_cleaned))
    
    for model_name, weight in zip(models_to_use, weights):
        if weight == 0:
            continue
            
        model_preds = np.zeros(len(test_cleaned))
        
        # Load and predict with 5 fold models and corresponding fold feature engineers
        for fold in range(5):
            fold_eng_path = f'outputs/models/feature_engineer_fold_{fold}.pkl'
            if not os.path.exists(fold_eng_path):
                raise FileNotFoundError(f"Missing fold engineer: {fold_eng_path}")
            with open(fold_eng_path, 'rb') as f:
                fold_engineer = pickle.load(f)
                
            test_feat_fold = fold_engineer.transform(test_cleaned)
            
            model_file = f'outputs/models/{model_name}_fold_{fold}.pkl'
            if not os.path.exists(model_file):
                raise FileNotFoundError(f"Missing fold model file: {model_file}")
                
            with open(model_file, 'rb') as f:
                fold_model = pickle.load(f)
                
            model_preds += fold_model.predict(test_feat_fold) / 5.0
            
        ensemble_preds += model_preds * weight
        
    final_prices = np.expm1(ensemble_preds)
    
    # Sanity checks
    assert not np.isnan(final_prices).any(), f"NaN values found in predictions!"
    assert (final_prices > 0).all(), f"Negative or zero prices predicted!"
    
    return final_prices

def main():
    print("Starting Inference Pipeline...")
    
    # 1. Load test data
    test_path = 'data/test.csv'
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test data at {test_path}")
        
    test_df = pd.read_csv(test_path)
    print(f"Loaded test data. Shape: {test_df.shape}")
    test_ids = test_df['Id']
    
    # 2. Preprocessing
    test_cleaned = clean_data(test_df, is_train=False)
    
    # 3. Save processed test features to data folder using global feature engineer (for reference / Tableau)
    feat_eng_path = 'outputs/models/feature_engineer.pkl'
    if os.path.exists(feat_eng_path):
        with open(feat_eng_path, 'rb') as f:
            global_engineer = pickle.load(f)
        test_feat_global = global_engineer.transform(test_cleaned)
        test_feat_with_id = test_feat_global.copy()
        test_feat_with_id['Id'] = test_ids
        test_feat_with_id.to_csv('data/test_processed.csv', index=False)
        print("Saved global test processed features for reference.")
    
    # 4. Load Ensemble Weights
    weights_path = 'outputs/models/ensemble_weights.csv'
    if not os.path.exists(weights_path):
        raise FileNotFoundError("Missing ensemble weights. Please run src/ensemble.py first.")
        
    weights_df = pd.read_csv(weights_path)
    models_to_use = weights_df['Model'].tolist()
    opt_weights = weights_df['Weight'].tolist()
    print(f"Loaded models: {models_to_use} with optimized weights: {opt_weights}")
    
    # Define weight scenarios for the 5 target submission types
    scenarios = {}
    
    # Scenario 1: Best Ensemble (SLSQP blend)
    scenarios['best_ensemble'] = opt_weights
    
    # Helper to build heavy blends
    def build_heavy_weights(heavy_model, heavy_fraction=0.7):
        new_weights = []
        other_sum = 0
        for m in models_to_use:
            if m != heavy_model:
                other_sum += opt_weights[models_to_use.index(m)]
        
        for m in models_to_use:
            if m == heavy_model:
                new_weights.append(heavy_fraction)
            else:
                orig_w = opt_weights[models_to_use.index(m)]
                # Distribute the remaining fraction proportionally
                if other_sum > 0:
                    new_weights.append((1.0 - heavy_fraction) * (orig_w / other_sum))
                else:
                    new_weights.append((1.0 - heavy_fraction) / (len(models_to_use) - 1))
        return new_weights

    # Scenario 2: CatBoost heavy (70% CatBoost)
    if 'CatBoost' in models_to_use:
        scenarios['catboost_heavy'] = build_heavy_weights('CatBoost', 0.7)
    
    # Scenario 3: XGBoost heavy (70% XGBoost)
    if 'XGBoost' in models_to_use:
        scenarios['xgboost_heavy'] = build_heavy_weights('XGBoost', 0.7)
        
    # Scenario 4: LightGBM heavy (70% LightGBM)
    if 'LightGBM' in models_to_use:
        scenarios['lightgbm_heavy'] = build_heavy_weights('LightGBM', 0.7)
        
    # Scenario 5: Stable Single (100% CatBoost)
    if 'CatBoost' in models_to_use:
        stable_w = [0.0] * len(models_to_use)
        stable_w[models_to_use.index('CatBoost')] = 1.0
        scenarios['stable_single'] = stable_w

    # Generate and save predictions for all scenarios
    for scenario_name, scenario_weights in scenarios.items():
        print(f"\nGenerating predictions for scenario: {scenario_name}...")
        print(f"Weights: {dict(zip(models_to_use, [round(w, 4) for w in scenario_weights]))}")
        
        prices = predict_blend(test_cleaned, models_to_use, scenario_weights)
        
        sub = pd.DataFrame({
            'Id': test_ids,
            'SalePrice': prices
        })
        
        sub_file = f'outputs/submissions/submission_{scenario_name}.csv'
        sub.to_csv(sub_file, index=False)
        print(f"Saved submission scenario to {sub_file}")
        
        # Also copy the best ensemble to the default submission.csv
        if scenario_name == 'best_ensemble':
            sub.to_csv('outputs/submissions/submission.csv', index=False)
            print("Copied best ensemble to outputs/submissions/submission.csv")

if __name__ == '__main__':
    main()
