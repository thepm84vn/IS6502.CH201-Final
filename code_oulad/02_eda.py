import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120

df = pd.read_csv('master_75pct.csv')  # use full-course view for demographic EDA
print(df.shape)
print(df.final_result.value_counts())

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# 1. Final result distribution
order = df.final_result.value_counts().index
sns.countplot(data=df, x='final_result', order=order, ax=axes[0,0], palette='viridis')
axes[0,0].set_title('Phân bố kết quả học tập cuối cùng (n=7,044)')
axes[0,0].set_xlabel(''); axes[0,0].set_ylabel('Số sinh viên')

# 2. Result by gender
sns.countplot(data=df, x='gender', hue='final_result', ax=axes[0,1], palette='viridis')
axes[0,1].set_title('Kết quả học tập theo giới tính')
axes[0,1].set_xlabel('Giới tính'); axes[0,1].set_ylabel('Số sinh viên')

# 3. Result by highest_education
edu_order = df.highest_education.value_counts().index
sns.countplot(data=df, y='highest_education', hue='final_result', ax=axes[1,0],
              order=edu_order, palette='viridis')
axes[1,0].set_title('Kết quả học tập theo trình độ học vấn trước đó')
axes[1,0].set_xlabel('Số sinh viên'); axes[1,0].set_ylabel('')

# 4. Total clicks (75%) by result
sns.boxplot(data=df, x='final_result', y='total_clicks_75', order=order, ax=axes[1,1], palette='viridis')
axes[1,1].set_title('Tổng lượt click VLE (đến mốc 75%) theo kết quả')
axes[1,1].set_ylabel('Tổng lượt click'); axes[1,1].set_xlabel('')
axes[1,1].set_yscale('log')

plt.tight_layout()
plt.savefig('eda_overview.png', bbox_inches='tight')
print("Saved eda_overview.png")

# Summary stats table for the paper
summary = df.groupby('final_result')[['total_clicks_75', 'active_days_75', 'avg_score_75']].mean().round(1)
print("\n=== Mean behavioral features by outcome (75% cutoff) ===")
print(summary)
summary.to_csv('summary_by_outcome.csv')

# Correlation of numeric behavioral features with outcome (encode pass-ish vs not)
df['pass_like'] = df.final_result.isin(['Pass', 'Distinction']).astype(int)
num_cols = ['total_clicks_75', 'active_days_75', 'n_materials_75', 'avg_score_75', 'n_submitted_75']
corr = df[num_cols + ['pass_like']].corr()['pass_like'].drop('pass_like').sort_values(ascending=False)
print("\n=== Correlation with Pass/Distinction outcome ===")
print(corr)
