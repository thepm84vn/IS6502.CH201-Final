import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report,
                              confusion_matrix)
from sklearn.preprocessing import LabelEncoder, label_binarize
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import json
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

CAT_COLS = ['gender', 'region', 'highest_education', 'imd_band', 'age_band',
            'disability', 'code_module', 'code_presentation']
NUM_COLS = ['num_of_prev_attempts', 'studied_credits', 'date_registration',
            'total_clicks', 'active_days', 'n_materials', 'first_access_day',
            'last_access_day', 'n_submitted', 'avg_score', 'n_banked']

results_all = {}

for cutoff in ['25', '50', '75']:
    print(f"\n{'='*60}\nCUTOFF = {cutoff}%\n{'='*60}")
    df = pd.read_csv(f'master_{cutoff}pct.csv')

    # rename cutoff-specific cols to generic names
    rename_map = {}
    for c in df.columns:
        if c.endswith(f'_{cutoff}'):
            rename_map[c] = c[: -(len(cutoff) + 1)]
    df = df.rename(columns=rename_map)

    df = df[df.final_result != 'Withdrawn'].copy()  # focus: Fail/Pass/Distinction for clarity
    # (Withdrawn kept out of the 3-class task here; a separate binary at-risk task
    #  including Withdrawn is reported via the "pass_like" analysis in EDA)

    for c in NUM_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df[NUM_COLS] = df[NUM_COLS].fillna(0)
    for c in CAT_COLS:
        df[c] = df[c].fillna('Unknown').astype(str)

    X = df[CAT_COLS + NUM_COLS]
    y = df['final_result']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_
    print("Classes:", classes, "| distribution:", dict(zip(*np.unique(y, return_counts=True))))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, random_state=42, stratify=y_enc)

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), CAT_COLS),
        ('num', StandardScaler(), NUM_COLS),
    ])

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # SMOTE to balance classes in training set only
    sm = SMOTE(random_state=42)
    X_train_bal, y_train_bal = sm.fit_resample(X_train_t, y_train)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=2000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        'SVM': SVC(probability=True, random_state=42, cache_size=1000),
        'XGBoost': xgb.XGBClassifier(n_estimators=300, random_state=42, eval_metric='mlogloss',
                                      use_label_encoder=False, n_jobs=-1),
    }

    # SVM training cost scales poorly (O(n^2)-O(n^3)) with probability=True (internal CV).
    # On the full OULAD dataset the balanced training set is too large for tractable SVM
    # training, so we cap SVM's training subsample while keeping all other models on 100%
    # of the (SMOTE-balanced) training data.
    SVM_MAX_TRAIN = 8000

    cutoff_results = {}
    for name, model in models.items():
        print(f"  -> training {name} ...", flush=True)
        if name == 'SVM' and X_train_bal.shape[0] > SVM_MAX_TRAIN:
            rng = np.random.RandomState(42)
            idx = rng.choice(X_train_bal.shape[0], SVM_MAX_TRAIN, replace=False)
            model.fit(X_train_bal[idx], y_train_bal[idx])
        else:
            model.fit(X_train_bal, y_train_bal)
        y_pred = model.predict(X_test_t)
        y_proba = model.predict_proba(X_test_t)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        try:
            y_test_bin = label_binarize(y_test, classes=np.unique(y_enc))
            auc = roc_auc_score(y_test_bin, y_proba, average='weighted', multi_class='ovr')
        except Exception as e:
            auc = np.nan

        cutoff_results[name] = {
            'accuracy': round(acc, 4), 'precision': round(prec, 4),
            'recall': round(rec, 4), 'f1': round(f1, 4), 'auc_roc': round(auc, 4) if not np.isnan(auc) else None
        }
        print(f"{name:20s} | Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} AUC={auc:.4f}")

    results_all[cutoff] = cutoff_results

    # Save best model artifacts at 75% for SHAP later
    if cutoff == '75':
        import joblib
        joblib.dump(models['XGBoost'], 'best_model_xgb_75.pkl')
        joblib.dump(preprocessor, 'preprocessor_75.pkl')
        np.save('X_test_75.npy', X_test_t if not hasattr(X_test_t, 'toarray') else X_test_t.toarray())
        X_test.to_csv('X_test_75_raw.csv', index=False)
        joblib.dump(le, 'label_encoder_75.pkl')
        # feature names
        feat_names = preprocessor.get_feature_names_out()
        with open('feature_names_75.json', 'w') as f:
            json.dump(list(feat_names), f)

with open('model_results_all_cutoffs.json', 'w') as f:
    json.dump(results_all, f, indent=2)

print("\n\n=== SUMMARY TABLE (best F1 per cutoff) ===")
for cutoff, res in results_all.items():
    best = max(res.items(), key=lambda kv: kv[1]['f1'])
    print(f"{cutoff}%: best model = {best[0]}, F1={best[1]['f1']}, Acc={best[1]['accuracy']}, AUC={best[1]['auc_roc']}")
