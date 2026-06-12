import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

def main():
    print("Starting SHAP Explainability Analysis...")
    
    # Create directories
    os.makedirs('outputs/feature_importance', exist_ok=True)
    
    # 1. Load preprocessed features
    processed_path = 'data/train_processed.csv'
    if not os.path.exists(processed_path):
        raise FileNotFoundError("Processed train data missing. Run src/train.py first.")
        
    df = pd.read_csv(processed_path)
    # Target columns
    y_log = df['SalePrice_log']
    X = df.drop(columns=['SalePrice_log', 'SalePrice'])
    
    # 2. Load a fold model (LightGBM is fast and compatible with TreeExplainer)
    model_path = 'outputs/models/LightGBM_fold_0.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file missing. Run src/train.py first.")
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    # 3. Create TreeExplainer
    explainer = shap.TreeExplainer(model)
    print("Calculating SHAP values...")
    shap_values = explainer(X)
    
    # 4. Generate summary plot (Global feature importance)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title('SHAP Feature Importance Summary (Top 20 Features)', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/feature_importance/shap_summary.png', dpi=300)
    plt.close()
    print("Generated shap_summary.png")
    
    # 5. Generate global importance bar plot
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.title('Global Mean Absolute SHAP Values', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/feature_importance/shap_bar.png', dpi=300)
    plt.close()
    print("Generated shap_bar.png")
    
    # 6. Generate waterfall plot for a single instance (Individual explanation)
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title('SHAP Explanation for House #1 Prediction', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/feature_importance/shap_individual.png', dpi=300)
    plt.close()
    print("Generated shap_individual.png")
    
if __name__ == '__main__':
    main()
