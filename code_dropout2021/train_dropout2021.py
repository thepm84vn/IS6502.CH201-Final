"""
Cross-dataset validation pipeline: UCI "Predict Students' Dropout and Academic
Success" (Realinho et al., 2021) -- a more recent (2021), different-institution,
different-country dataset used to validate whether the early-prediction
methodology developed on OULAD (2013-2014, UK) generalizes.

Two natural checkpoints (mirroring OULAD's 25/50/75% early-prediction design):
  - Checkpoint 1: after 1st semester only (demographics + 1st-sem curricular data)
  - Checkpoint 2: after 2nd semester (demographics + full-year curricular data)
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, label_binarize
from sklearn.compose import ColumnTransformer
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
df.columns = [c.strip().lstrip('\ufeff') for c in df.columns]
print("Shape:", df.shape)
print(df['Target'].value_counts())

DEMOGRAPHIC_COLS = [
    'Marital status', 'Application mode', 'Application order', 'Course',
    'Daytime/evening attendance', 'Previous qualification', 'Nacionality',
    "Mother's qualification", "Father's qualification", "Mother's occupation",
    "Father's occupation", 'Displaced', 'Educational special needs', 'Debtor',
    'Tuition fees up to date', 'Gender', 'Scholarship holder', 'Age at enrollment',
    'International', 'Unemployment rate', 'Inflation rate', 'GDP',
]
SEM1_COLS = [c for c in df.columns if '1st sem' in c]
SEM2_COLS = [c for c in df.columns if '2nd sem' in c]

CHECKPOINTS = {
    'sem1': DEMOGRAPHIC_COLS + SEM1_COLS,
    'sem2': DEMOGRAPHIC_COLS + SEM1_COLS + SEM2_COLS,
}

le = LabelEncoder()
y_all = le.fit_transform(df['Target'])
print("Classes:", le.classes_)

results_all = {}
saved = {}

for cp_name, feat_cols in CHECKPOINTS.items():
    print(f"\n{'='*60}\nCHECKPOINT = {cp_name} ({len(feat_cols)} features)\n{'='*60}")
    X = df[feat_cols].copy()
    y = y_all

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_t = scaler.fit_transform(X_train)
    X_test_t = scaler.transform(X_test)

    sm = SMOTE(random_state=42)
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
        y_test_bin = label_binarize(y_test, classes=np.unique(y_all))
        auc = roc_auc_score(y_test_bin, y_proba, average='weighted', multi_class='ovr')

        cp_results[name] = {'accuracy': round(acc, 4), 'precision': round(prec, 4),
                             'recall': round(rec, 4), 'f1': round(f1, 4), 'auc_roc': round(auc, 4)}
        print(f"{name:20s} | Acc={acc:.4f} F1={f1:.4f} AUC={auc:.4f}")

        if cp_name == 'sem2' and name == 'XGBoost':
            joblib.dump(model, 'best_model_xgb_sem2.pkl')
            joblib.dump(scaler, 'scaler_sem2.pkl')
            joblib.dump(le, 'label_encoder.pkl')
            np.save('X_test_sem2.npy', X_test_t)
            with open('feature_names_sem2.json', 'w') as f:
                json.dump(feat_cols, f)
            cm = confusion_matrix(y_test, y_pred)
            np.save('confusion_matrix_sem2.npy', cm)

    results_all[cp_name] = cp_results

with open('model_results_dropout2021.json', 'w') as f:
    json.dump(results_all, f, indent=2)

print("\n=== SUMMARY ===")
for cp, res in results_all.items():
    best = max(res.items(), key=lambda kv: kv[1]['f1'])
    print(f"{cp}: best = {best[0]}, F1={best[1]['f1']}, Acc={best[1]['accuracy']}, AUC={best[1]['auc_roc']}")

# ---- Early-prediction trend chart ----
plt.figure(figsize=(7.5, 5.3))
labels_map = {'sem1': 'Sau HK1', 'sem2': 'Sau HK2 (đầy đủ)'}
for m in ['LogisticRegression', 'RandomForest', 'SVM', 'XGBoost']:
    vals = [results_all[cp][m]['f1'] for cp in ['sem1', 'sem2']]
    plt.plot([labels_map['sem1'], labels_map['sem2']], vals, marker='o', linewidth=2, label=m)
plt.xlabel('Mốc thời gian (Dataset 2021)')
plt.ylabel('F1-score (weighted)')
plt.title('Hiệu suất dự đoán sớm — Dataset đối chứng 2021 (Bồ Đào Nha)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('dropout2021_early_prediction_trend.png', dpi=120, bbox_inches='tight')
print("Saved dropout2021_early_prediction_trend.png")

# ---- Confusion matrix chart ----
cm = np.load('confusion_matrix_sem2.npy')
plt.figure(figsize=(6, 5.2))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Dự đoán'); plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn — XGBoost, HK2 đầy đủ (Dataset 2021)')
plt.tight_layout()
plt.savefig('dropout2021_confusion_matrix.png', dpi=120, bbox_inches='tight')
print("Saved dropout2021_confusion_matrix.png")

# ---- SHAP ----
model = joblib.load('best_model_xgb_sem2.pkl')
X_test = np.load('X_test_sem2.npy')
with open('feature_names_sem2.json') as f:
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
print("\n=== TOP 15 FEATURES (Dataset 2021) ===")
print(importance.head(15))
importance.head(20).to_csv('shap_top20_dropout2021.csv')

plt.figure(figsize=(9, 7))
top15 = importance.head(15).sort_values()
plt.barh(top15.index, top15.values, color='#2C5F2D')
plt.xlabel('Mean |SHAP value|')
plt.title('Top 15 đặc trưng quan trọng nhất — Dataset 2021 (XGBoost, HK2)')
plt.tight_layout()
plt.savefig('dropout2021_shap_importance.png', dpi=120, bbox_inches='tight')
print("Saved dropout2021_shap_importance.png")
