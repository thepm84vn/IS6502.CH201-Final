"""
Third validation dataset: North American university course (Injadat et al., 2020,
Applied Intelligence) -- adds a third independent geographic/institutional context
(UK-OULAD 2013-14, Portugal 2021, North America ~2020) to the early-prediction
methodology, using the dataset's native checkpoints: 20% and 50% of coursework.
"""
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

np.random.seed(42)
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120

df = pd.read_csv('dataset.csv')
print("Shape:", df.shape)
print(df['Class'].value_counts())

# ---- Checkpoints mirroring the dataset's native design (20% / 50% of coursework) ----
CHECKPOINTS = {
    'pct20': ['Quiz01 [10]', 'Assignment01 [8]'],                                   # ~20% of course
    'pct50': ['Quiz01 [10]', 'Assignment01 [8]', 'Midterm Exam [20]', 'Assignment02 [12]'],  # ~50%
}

le = LabelEncoder()
y_all = le.fit_transform(df['Class'])
print("Classes:", le.classes_, "counts:", np.bincount(y_all))

results_all = {}

for cp_name, feat_cols in CHECKPOINTS.items():
    print(f"\n{'='*60}\nCHECKPOINT = {cp_name} ({feat_cols})\n{'='*60}")
    X = df[feat_cols].copy()
    y = y_all

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_t = scaler.fit_transform(X_train)
    X_test_t = scaler.transform(X_test)

    # SMOTE: k_neighbors must be < smallest class count in training data
    min_class_count = pd.Series(y_train).value_counts().min()
    k_neighbors = max(1, min(5, min_class_count - 1))
    sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_train_bal, y_train_bal = sm.fit_resample(X_train_t, y_train)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=2000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        'SVM': SVC(probability=True, random_state=42, cache_size=1000),
        'XGBoost': xgb.XGBClassifier(n_estimators=300, random_state=42, eval_metric='mlogloss',
                                      use_label_encoder=False, n_jobs=-1),
    }

    cp_results = {}
    for name, model in models.items():
        print(f"  -> training {name} ...", flush=True)
        model.fit(X_train_bal, y_train_bal)
        y_pred = model.predict(X_test_t)
        y_proba = model.predict_proba(X_test_t)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        try:
            y_test_bin = label_binarize(y_test, classes=np.unique(y_all))
            auc = roc_auc_score(y_test_bin, y_proba, average='weighted', multi_class='ovr')
        except Exception:
            auc = np.nan

        cp_results[name] = {'accuracy': round(acc, 4), 'precision': round(prec, 4),
                             'recall': round(rec, 4), 'f1': round(f1, 4),
                             'auc_roc': round(auc, 4) if not np.isnan(auc) else None}
        print(f"{name:20s} | Acc={acc:.4f} F1={f1:.4f} AUC={auc:.4f}")

        if cp_name == 'pct50' and name == 'XGBoost':
            joblib.dump(model, 'best_model_xgb_pct50.pkl')
            joblib.dump(scaler, 'scaler_pct50.pkl')
            joblib.dump(le, 'label_encoder.pkl')
            np.save('X_test_pct50.npy', X_test_t)
            with open('feature_names_pct50.json', 'w') as f:
                json.dump(feat_cols, f)
            cm = confusion_matrix(y_test, y_pred)
            np.save('confusion_matrix_pct50.npy', cm)

    results_all[cp_name] = cp_results

with open('model_results_namerica.json', 'w') as f:
    json.dump(results_all, f, indent=2)

print("\n=== SUMMARY ===")
for cp, res in results_all.items():
    best = max(res.items(), key=lambda kv: kv[1]['f1'])
    print(f"{cp}: best = {best[0]}, F1={best[1]['f1']}, Acc={best[1]['accuracy']}, AUC={best[1]['auc_roc']}")

# ---- Early-prediction trend chart ----
plt.figure(figsize=(7.5, 5.3))
labels_map = {'pct20': 'Mốc 20%', 'pct50': 'Mốc 50%'}
for m in ['LogisticRegression', 'RandomForest', 'SVM', 'XGBoost']:
    vals = [results_all[cp][m]['f1'] for cp in ['pct20', 'pct50']]
    plt.plot([labels_map['pct20'], labels_map['pct50']], vals, marker='o', linewidth=2, label=m)
plt.xlabel('Mốc thời gian (Dataset Bắc Mỹ)')
plt.ylabel('F1-score (weighted)')
plt.title('Hiệu suất dự đoán sớm — Dataset đối chứng Bắc Mỹ')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('namerica_early_prediction_trend.png', dpi=120, bbox_inches='tight')
print("Saved namerica_early_prediction_trend.png")

# ---- Confusion matrix chart ----
cm = np.load('confusion_matrix_pct50.npy')
plt.figure(figsize=(6, 5.2))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Dự đoán'); plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn — XGBoost, mốc 50% (Dataset Bắc Mỹ)')
plt.tight_layout()
plt.savefig('namerica_confusion_matrix.png', dpi=120, bbox_inches='tight')
print("Saved namerica_confusion_matrix.png")

# ---- SHAP ----
model = joblib.load('best_model_xgb_pct50.pkl')
X_test = np.load('X_test_pct50.npy')
with open('feature_names_pct50.json') as f:
    feat_names = json.load(f)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
if isinstance(shap_values, list):
    mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)
elif shap_values.ndim == 3:
    mean_abs = np.abs(shap_values).mean(axis=(0, 2))
else:
    mean_abs = np.abs(shap_values).mean(axis=0)

importance = pd.Series(mean_abs, index=feat_names).sort_values(ascending=False)
print("\n=== FEATURES (Dataset Bắc Mỹ, mốc 50%) ===")
print(importance)
importance.to_csv('shap_namerica.csv')

plt.figure(figsize=(8, 4.5))
top = importance.sort_values()
plt.barh(top.index, top.values, color='#6A4C93')
plt.xlabel('Mean |SHAP value|')
plt.title('Đặc trưng quan trọng nhất — Dataset Bắc Mỹ (XGBoost, mốc 50%)')
plt.tight_layout()
plt.savefig('namerica_shap_importance.png', dpi=120, bbox_inches='tight')
print("Saved namerica_shap_importance.png")
