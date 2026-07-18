import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix
import seaborn as sns

with open('model_results_all_cutoffs.json') as f:
    results = json.load(f)

# --- Chart 1: Early prediction trend ---
cutoffs = ['25', '50', '75']
models_order = ['LogisticRegression', 'RandomForest', 'SVM', 'XGBoost']
plt.figure(figsize=(8, 5.5))
for m in models_order:
    accs = [results[c][m]['f1'] for c in cutoffs]
    plt.plot(['25%', '50%', '75%'], accs, marker='o', linewidth=2, label=m)
plt.xlabel('Mốc thời gian trong khóa học (% thời lượng đã qua)')
plt.ylabel('F1-score (weighted)')
plt.title('Hiệu suất dự đoán theo mốc thời gian sớm (Early Prediction)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('early_prediction_trend.png', dpi=120, bbox_inches='tight')
print("Saved early_prediction_trend.png")

# --- Chart 2: Confusion matrix for best model (XGBoost @ 75%) ---
model = joblib.load('best_model_xgb_75.pkl')
X_test = np.load('X_test_75.npy')
le = joblib.load('label_encoder_75.pkl')

# need y_test - reload from raw pipeline quickly using same split logic
df = pd.read_csv('master_75pct.csv')
rename_map = {c: c[:-3] for c in df.columns if c.endswith('_75')}
df = df.rename(columns=rename_map)
df = df[df.final_result != 'Withdrawn'].copy()

from sklearn.model_selection import train_test_split
CAT_COLS = ['gender', 'region', 'highest_education', 'imd_band', 'age_band',
            'disability', 'code_module', 'code_presentation']
NUM_COLS = ['num_of_prev_attempts', 'studied_credits', 'date_registration',
            'total_clicks', 'active_days', 'n_materials', 'first_access_day',
            'last_access_day', 'n_submitted', 'avg_score', 'n_banked']
for c in NUM_COLS:
    if c not in df.columns: df[c] = np.nan
df[NUM_COLS] = df[NUM_COLS].fillna(0)
for c in CAT_COLS: df[c] = df[c].fillna('Unknown').astype(str)
y_enc = le.transform(df['final_result'])
X = df[CAT_COLS + NUM_COLS]
_, _, y_train, y_test = train_test_split(X, y_enc, test_size=0.25, random_state=42, stratify=y_enc)

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(6.5, 5.5))
sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Dự đoán'); plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn — XGBoost (mốc 75%)')
plt.tight_layout()
plt.savefig('confusion_matrix_75.png', dpi=120, bbox_inches='tight')
print("Saved confusion_matrix_75.png")

print("\ny_test distribution:", np.unique(y_test, return_counts=True))
print("Accuracy check:", (y_pred == y_test).mean())
