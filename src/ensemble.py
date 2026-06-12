import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    print("Starting Ensemble Optimization...")
    oof_path = 'outputs/oof_predictions.csv'
    if not os.path.exists(oof_path):
        raise FileNotFoundError("Please run src/train.py first to generate OOF predictions.")
        
    oof_df = pd.read_csv(oof_path)
    target = oof_df['Target'].values
    
    # Models to ensemble
    # We will prioritize the baseline models: ElasticNet, LightGBM, CatBoost, but we can also use XGBoost.
    model_names = ['CatBoost', 'LightGBM', 'ElasticNet', 'XGBoost']
    
    # Check if models are in the OOF DataFrame
    available_models = [m for m in model_names if m in oof_df.columns]
    print(f"Available models for ensembling: {available_models}")
    
    X_preds = oof_df[available_models].values
    
    # 1. Evaluate default prompt weights if available
    # Default: 0.4 * CatBoost + 0.3 * LightGBM + 0.3 * ElasticNet
    if 'CatBoost' in available_models and 'LightGBM' in available_models and 'ElasticNet' in available_models:
        w_default = np.zeros(len(available_models))
        w_default[available_models.index('CatBoost')] = 0.4
        w_default[available_models.index('LightGBM')] = 0.3
        w_default[available_models.index('ElasticNet')] = 0.3
        
        pred_default = np.dot(X_preds, w_default)
        score_default = rmse(target, pred_default)
        print(f"Default Blend (0.4 CatBoost + 0.3 LightGBM + 0.3 ElasticNet) OOF RMSLE: {score_default:.5f}")
        
    # 2. Run optimization to find optimal weights
    def loss_func(weights):
        # Normalize weights to sum to 1
        w = weights / np.sum(weights)
        pred = np.dot(X_preds, w)
        return rmse(target, pred)
        
    init_weights = np.ones(len(available_models)) / len(available_models)
    bounds = [(0, 1)] * len(available_models)
    constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    
    res = minimize(loss_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    opt_weights = res.x
    opt_weights = opt_weights / np.sum(opt_weights) # Ensure strict normalization
    
    opt_pred = np.dot(X_preds, opt_weights)
    opt_score = rmse(target, opt_pred)
    
    print("\nOptimized Ensemble Weights:")
    for m, w in zip(available_models, opt_weights):
        print(f"  - {m}: {w:.4f}")
        
    print(f"Optimized Ensemble OOF RMSLE: {opt_score:.5f}")
    
    # Save optimized weights to a json or csv file
    weights_df = pd.DataFrame({
        'Model': available_models,
        'Weight': opt_weights
    })
    weights_df.to_csv('outputs/models/ensemble_weights.csv', index=False)
    print("Saved optimized weights to outputs/models/ensemble_weights.csv")
    
    # Log the optimized ensemble experiment
    exp_file = 'outputs/experiments.csv'
    if os.path.exists(exp_file):
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        exp_row = pd.DataFrame([{
            'Timestamp': timestamp,
            'Model': 'Optimized_Ensemble',
            'CV_RMSLE': round(opt_score, 5),
            'FeatureSet_Version': 'v1.0 (16 Engineered Features)',
            'Hyperparameters': str({m: round(w, 4) for m, w in zip(available_models, opt_weights)}),
            'Notes': 'Nelder-Mead/SLSQP optimization of out-of-fold predictions'
        }])
        old_df = pd.read_csv(exp_file)
        new_df = pd.concat([old_df, exp_row], ignore_index=True)
        new_df.to_csv(exp_file, index=False)

if __name__ == '__main__':
    main()
