"""
Fourth validation dataset: xAPI-Edu-Data (Kalboard 360 LMS, Amman/Kuwait,
collected 2016; Amrieh, Hamtini & Aljarah). This dataset is the closest in SPIRIT
to OULAD among all our validation sets: it contains genuine online-interaction
behavioral features (raisedhands, VisITedResources, AnnouncementsView, Discussion)
rather than only exam/curricular scores -- directly analogous to OULAD's clickstream.

It adds a 4th independent geographic context (UK, Portugal, N.America, Middle East)
and lets us test the "behavior-only early prediction" idea in its purest form:
  - Checkpoint 1 (behavior-only): only the 4 interaction features -- the OULAD-style
    "can we predict from engagement behavior alone?" question.
  - Checkpoint 2 (full): behavior features + demographic/contextual features.
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
print("Shape:", df.shape)
print(df['Class'].value_counts())

BEHAVIOR_COLS = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']
CONTEXT_CAT = ['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID',
               'SectionID', 'Topic', 'Semester', 'Relation', 'ParentAnsweringSurvey',
               'ParentschoolSatisfaction', 'StudentAbsenceDays']

# Two checkpoints
CHECKPOINTS = {
    'behavior_only': (BEHAVIOR_COLS, []),
    'full': (BEHAVIOR_COLS, CONTEXT_CAT),
}

le = LabelEncoder()
# Order classes L < M < H for interpretability
df['Class'] = pd.Categorical(df['Class'], categories=['L', 'M', 'H'], ordered=True)
y_all = le.fit_transform(df['Class'].astype(str))
print("Classes:", le.classes_, "counts:", np.bincount(y_all))

results_all = {}

for cp_name, (num_cols, cat_cols) in CHECKPOINTS.items():
    print(f"\n{'='*60}\nCHECKPOINT = {cp_name}\n{'='*60}")
    X = df[num_cols + cat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype(str)
    y = y_all

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    if cat_cols:
        pre = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ])
    else:
        pre = ColumnTransformer([('num', StandardScaler(), num_cols)])

    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)
    if hasattr(X_train_t, 'toarray'):
        X_train_t = X_train_t.toarray(); X_test_t = X_test_t.toarray()

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

        if cp_name == 'full' and name == 'XGBoost':
            joblib.dump(model, 'best_model_xgb_full.pkl')
            joblib.dump(pre, 'preprocessor_full.pkl')
            joblib.dump(le, 'label_encoder.pkl')
            np.save('X_test_full.npy', X_test_t)
            feat_names = list(pre.get_feature_names_out())
            with open('feature_names_full.json', 'w') as f:
                json.dump(feat_names, f)
            cm = confusion_matrix(y_test, y_pred)
            np.save('confusion_matrix_full.npy', cm)

    results_all[cp_name] = cp_results

with open('model_results_xapi.json', 'w') as f:
    json.dump(results_all, f, indent=2)

print("\n=== SUMMARY ===")
for cp, res in results_all.items():
    best = max(res.items(), key=lambda kv: kv[1]['f1'])
    print(f"{cp}: best = {best[0]}, F1={best[1]['f1']}, Acc={best[1]['accuracy']}, AUC={best[1]['auc_roc']}")

# ---- Trend chart ----
plt.figure(figsize=(7.5, 5.3))
labels_map = {'behavior_only': 'Chỉ hành vi\ntương tác', 'full': 'Hành vi +\nbối cảnh đầy đủ'}
for m in ['LogisticRegression', 'RandomForest', 'SVM', 'XGBoost']:
    vals = [results_all[cp][m]['f1'] for cp in ['behavior_only', 'full']]
    plt.plot([labels_map['behavior_only'], labels_map['full']], vals, marker='o', linewidth=2, label=m)
plt.xlabel('Bộ đặc trưng (Dataset xAPI-Edu, Trung Đông)')
plt.ylabel('F1-score (weighted)')
plt.title('Hiệu suất dự đoán — Dataset đối chứng xAPI-Edu')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('xapi_early_prediction_trend.png', dpi=120, bbox_inches='tight')
print("Saved xapi_early_prediction_trend.png")

# ---- Confusion matrix ----
cm = np.load('confusion_matrix_full.npy')
plt.figure(figsize=(6, 5.2))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Dự đoán'); plt.ylabel('Thực tế')
plt.title('Ma trận nhầm lẫn — XGBoost, đầy đủ (Dataset xAPI-Edu)')
plt.tight_layout()
plt.savefig('xapi_confusion_matrix.png', dpi=120, bbox_inches='tight')
print("Saved xapi_confusion_matrix.png")

# ---- SHAP ----
model = joblib.load('best_model_xgb_full.pkl')
X_test = np.load('X_test_full.npy')
with open('feature_names_full.json') as f:
    feat_names = json.load(f)
clean = [n.replace('num__', '').replace('cat__', '') for n in feat_names]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
if isinstance(shap_values, list):
    mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)
elif shap_values.ndim == 3:
    mean_abs = np.abs(shap_values).mean(axis=(0, 2))
else:
    mean_abs = np.abs(shap_values).mean(axis=0)

importance = pd.Series(mean_abs, index=clean).sort_values(ascending=False)
print("\n=== TOP 15 FEATURES (xAPI-Edu) ===")
print(importance.head(15))
importance.head(20).to_csv('shap_xapi.csv')

plt.figure(figsize=(9, 6))
top = importance.head(12).sort_values()
plt.barh(top.index, top.values, color='#D35400')
plt.xlabel('Mean |SHAP value|')
plt.title('Top 12 đặc trưng quan trọng nhất — Dataset xAPI-Edu (XGBoost)')
plt.tight_layout()
plt.savefig('xapi_shap_importance.png', dpi=120, bbox_inches='tight')
print("Saved xapi_shap_importance.png")
