import joblib
import numpy as np
import pandas as pd
import json
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

model = joblib.load('best_model_xgb_75.pkl')
X_test = np.load('X_test_75.npy')
with open('feature_names_75.json') as f:
    feat_names = json.load(f)
le = joblib.load('label_encoder_75.pkl')

# Clean up feature names for readability
clean_names = [n.replace('cat__', '').replace('num__', '') for n in feat_names]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print("SHAP values type:", type(shap_values), "shape:", np.array(shap_values).shape if not isinstance(shap_values, list) else [s.shape for s in shap_values])

# multiclass -> shap_values shape (n_samples, n_features, n_classes) for newer shap, or list of arrays for older
X_test_df = pd.DataFrame(X_test, columns=clean_names)

if isinstance(shap_values, list):
    mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)
elif shap_values.ndim == 3:
    mean_abs = np.abs(shap_values).mean(axis=(0, 2))
else:
    mean_abs = np.abs(shap_values).mean(axis=0)

importance = pd.Series(mean_abs, index=clean_names).sort_values(ascending=False)
print("\n=== TOP 15 FEATURES (mean |SHAP value|) ===")
print(importance.head(15))
importance.head(20).to_csv('shap_top20.csv')

# Plot
plt.figure(figsize=(9, 7))
top15 = importance.head(15).sort_values()
plt.barh(top15.index, top15.values, color='#4C72B0')
plt.xlabel('Mean |SHAP value| (mức độ ảnh hưởng trung bình)')
plt.title('Top 15 đặc trưng quan trọng nhất — Mô hình XGBoost (mốc 75%)')
plt.tight_layout()
plt.savefig('shap_importance.png', bbox_inches='tight', dpi=120)
print("\nSaved shap_importance.png")
