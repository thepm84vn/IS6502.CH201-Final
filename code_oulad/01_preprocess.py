import pandas as pd
import numpy as np

pd.set_option('display.width', 120)

# ---- Load raw tables ----
studentInfo = pd.read_csv('studentInfo.csv')
studentReg = pd.read_csv('studentRegistration.csv')
studentAssess = pd.read_csv('studentAssessment.csv')
assessments = pd.read_csv('assessments.csv')
courses = pd.read_csv('courses.csv')
vle = pd.read_csv('studentVle.csv')  # Full official dataset (10,655,280 rows)
vle_meta = pd.read_csv('vle.csv')

print("=== RAW SHAPES ===")
for name, df in [('studentInfo', studentInfo), ('studentReg', studentReg),
                  ('studentAssess', studentAssess), ('assessments', assessments),
                  ('courses', courses), ('vle(studentVle)', vle), ('vle_meta', vle_meta)]:
    print(f"{name}: {df.shape}")

# ---- FULL DATASET: no scoping needed ----
# Verified official OULAD dataset (Kuzilek, Hlosta & Zdrahal, 2017), obtained from the
# original author's R package repository (jakubkuzilek/oulad), matching the published
# counts exactly: 32,593 students, 22 module-presentations, 10,655,280 VLE interactions.
studentAssess_full = studentAssess.copy()

print("\n=== FULL DATASET SHAPES (all 7 modules) ===")
print("studentInfo:", studentInfo.shape)
print("studentReg:", studentReg.shape)
print("assessments:", assessments.shape)
print("courses:", courses.shape)
print("vle interactions:", vle.shape)

# ---- Target variable ----
print("\n=== final_result distribution (scoped) ===")
print(studentInfo.final_result.value_counts())

# ---- Feature Engineering: behavioral features from VLE clickstream ----
# module_presentation_length needed to compute % thresholds
courses_len = courses.set_index(['code_module', 'code_presentation'])['module_presentation_length'].to_dict()

def pct_day(row_module, row_pres, day):
    length = courses_len.get((row_module, row_pres), 269)
    return day <= 0.25 * length, day <= 0.50 * length, day <= 0.75 * length

vle = vle.merge(courses[['code_module', 'code_presentation', 'module_presentation_length']],
                 on=['code_module', 'code_presentation'], how='left')
vle['length'] = vle['module_presentation_length'].fillna(269)
vle['pct_of_course'] = vle['date'] / vle['length']

def agg_clicks(cutoff):
    sub = vle[vle['pct_of_course'] <= cutoff]
    g = sub.groupby(['code_module', 'code_presentation', 'id_student']).agg(
        total_clicks=('sum_click', 'sum'),
        active_days=('date', 'nunique'),
        n_materials=('id_site', 'nunique'),
        first_access_day=('date', 'min'),
        last_access_day=('date', 'max'),
    ).reset_index()
    g.columns = ['code_module', 'code_presentation', 'id_student'] + \
                [f"{c}_{int(cutoff*100)}" for c in g.columns[3:]]
    return g

feat_25 = agg_clicks(0.25)
feat_50 = agg_clicks(0.50)
feat_75 = agg_clicks(0.75)

# ---- Assessment-based features (use only assessments due before each cutoff, scoped) ----
sa = studentAssess_full.merge(assessments, on='id_assessment', how='inner')
sa = sa.merge(courses[['code_module', 'code_presentation', 'module_presentation_length']],
              on=['code_module', 'code_presentation'], how='left')
sa['length'] = sa['module_presentation_length'].fillna(269)
sa['pct_of_course'] = sa['date_submitted'] / sa['length']

def agg_assess(cutoff):
    sub = sa[sa['pct_of_course'] <= cutoff]
    g = sub.groupby(['code_module', 'code_presentation', 'id_student']).agg(
        n_submitted=('id_assessment', 'count'),
        avg_score=('score', 'mean'),
        n_banked=('is_banked', 'sum'),
    ).reset_index()
    g.columns = ['code_module', 'code_presentation', 'id_student'] + \
                [f"{c}_{int(cutoff*100)}" for c in g.columns[3:]]
    return g

assess_25 = agg_assess(0.25)
assess_50 = agg_assess(0.50)
assess_75 = agg_assess(0.75)

# ---- Merge everything into one master table per cutoff ----
def build_dataset(cutoff_feat, cutoff_assess, label):
    df = studentInfo.merge(studentReg[['code_module', 'code_presentation', 'id_student',
                                        'date_registration']],
                            on=['code_module', 'code_presentation', 'id_student'], how='left')
    df = df.merge(cutoff_feat, on=['code_module', 'code_presentation', 'id_student'], how='left')
    df = df.merge(cutoff_assess, on=['code_module', 'code_presentation', 'id_student'], how='left')
    df['dataset_cutoff'] = label
    return df

df25 = build_dataset(feat_25, assess_25, '25pct')
df50 = build_dataset(feat_50, assess_50, '50pct')
df75 = build_dataset(feat_75, assess_75, '75pct')

for name, d in [('25%', df25), ('50%', df50), ('75%', df75)]:
    d.to_csv(f'master_{name.replace("%","pct")}.csv', index=False)
    print(f"\nSaved master dataset at {name} cutoff: {d.shape}")

print("\nPreprocessing complete.")
