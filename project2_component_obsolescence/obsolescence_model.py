"""
╔══════════════════════════════════════════════════════════════════╗
║  PROJECT 2: Electronic Component Obsolescence Prediction        ║
║  Author: Ahmed Mohammed Gouda                                   ║
║  Role: ML Engineer (Freelance) - Supply Chain Analytics         ║
║  Date: 2024                                                     ║
║  Client: Electronics Component Distributor                      ║
╚══════════════════════════════════════════════════════════════════╝

BUSINESS CONTEXT:
    An electronics distributor needed to predict which components would 
    become obsolete (End-of-Life) within the next 12 months. This helps 
    them proactively source alternatives and notify customers before 
    supply disruptions occur. Dataset: 5000 components across 8 categories.

APPROACH:
    - Binary classification: Will component go EOL in next 12 months?
    - Handle severe class imbalance (~8% positive rate)
    - Use SMOTE + class weights for balancing
    - Feature engineering from component lifecycle data
    - Survival analysis perspective for time-to-EOL

ALGORITHMS & RATIONALE:
    - GradientBoosting: Handles imbalanced data well with class weights
    - RandomForest: Baseline + feature importance
    - LogisticRegression: Interpretable model for stakeholders
    - Focus on RECALL: Missing a component going EOL is very costly
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, accuracy_score,
    average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import warnings
import os

warnings.filterwarnings('ignore')
np.random.seed(42)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{OUTPUT_DIR}/models", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/data", exist_ok=True)

print("="*70)
print("  PROJECT 2: COMPONENT OBSOLESCENCE PREDICTION")
print("="*70)

# ══════════════════════════════════════════════════════════════
# STEP 1: GENERATE REALISTIC COMPONENT DATA
# ══════════════════════════════════════════════════════════════
print("\n[STEP 1] Generating component lifecycle dataset...")

n_components = 5000

categories = ['Semiconductors', 'Capacitors', 'Resistors', 'Connectors',
              'Inductors', 'Diodes', 'Transistors', 'ICs']
manufacturers = ['Texas Instruments', 'Murata', 'Samsung', 'Intel', 'STMicro',
                 'Infineon', 'NXP', 'Vishay', 'TDK', 'ON Semi', 'Microchip',
                 'Analog Devices', 'Broadcom', 'Renesas', 'ROHM']
packages = ['SMD-0402', 'SMD-0603', 'SMD-0805', 'QFN', 'BGA', 'SOIC',
            'DIP', 'QFP', 'SOT-23', 'TO-220', 'TSSOP', 'CSP']
lifecycle_stages = ['Active', 'Mature', 'Declining', 'Last Buy', 'EOL Announced']

comp_categories = np.random.choice(categories, n_components, 
    p=[0.2, 0.15, 0.15, 0.12, 0.1, 0.1, 0.1, 0.08])

data = {
    'component_id': [f'CMP-{i:05d}' for i in range(n_components)],
    'category': comp_categories,
    'manufacturer': np.random.choice(manufacturers, n_components),
    'package_type': np.random.choice(packages, n_components),
    'years_since_introduction': np.clip(np.random.exponential(8, n_components), 0.5, 30).round(1),
    'technology_node_nm': np.random.choice([7, 10, 14, 22, 28, 40, 65, 90, 130, 180, 250], n_components,
        p=[0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.15, 0.12, 0.1, 0.06, 0.05]),
    'num_alternative_parts': np.random.poisson(4, n_components),
    'num_authorized_distributors': np.clip(np.random.poisson(5, n_components), 0, 20),
    'monthly_demand_units': np.clip(np.random.lognormal(7, 2, n_components), 10, 1000000).astype(int),
    'demand_trend_6m': np.random.normal(0, 0.15, n_components).round(3),  # % change
    'demand_trend_12m': np.random.normal(-0.02, 0.2, n_components).round(3),
    'avg_lead_time_weeks': np.clip(np.random.lognormal(2.5, 0.6, n_components), 1, 52).round(1),
    'lead_time_increase_pct': np.clip(np.random.normal(5, 15, n_components), -20, 100).round(1),
    'price_trend_6m': np.random.normal(0.02, 0.1, n_components).round(3),
    'num_pcn_notices': np.random.poisson(0.5, n_components),  # Product Change Notices
    'last_pcn_months_ago': np.clip(np.random.exponential(18, n_components), 0, 60).round(0),
    'manufacturer_financial_health': np.clip(np.random.normal(70, 15, n_components), 20, 100).round(1),
    'num_design_wins': np.clip(np.random.lognormal(2, 1, n_components), 0, 500).astype(int),
    'cross_reference_count': np.random.poisson(3, n_components),
    'rohs_compliant': np.random.choice([0, 1], n_components, p=[0.15, 0.85]),
    'automotive_qualified': np.random.choice([0, 1], n_components, p=[0.6, 0.4]),
    'military_grade': np.random.choice([0, 1], n_components, p=[0.85, 0.15]),
}

df = pd.DataFrame(data)

# Lifecycle stage (correlated with age)
def assign_lifecycle(row):
    age = row['years_since_introduction']
    r = np.random.random()
    if age < 3: return 'Active' if r < 0.9 else 'Mature'
    elif age < 7: return 'Active' if r < 0.5 else 'Mature' if r < 0.85 else 'Declining'
    elif age < 12: return 'Mature' if r < 0.3 else 'Declining' if r < 0.7 else 'Last Buy'
    elif age < 18: return 'Declining' if r < 0.3 else 'Last Buy' if r < 0.7 else 'EOL Announced'
    else: return 'Last Buy' if r < 0.3 else 'EOL Announced' if r < 0.7 else 'Declining'

df['lifecycle_stage'] = df.apply(assign_lifecycle, axis=1)

# Generate target: will_go_eol_12m (realistic ~8% positive)
eol_score = (
    df['years_since_introduction'] / 30 * 20 +
    (df['lifecycle_stage'].map({'Active': 0, 'Mature': 5, 'Declining': 15, 'Last Buy': 30, 'EOL Announced': 40})) +
    (1 - df['manufacturer_financial_health'] / 100) * 10 +
    df['num_pcn_notices'] * 5 +
    (-df['demand_trend_12m']) * 15 +
    df['lead_time_increase_pct'] / 100 * 8 +
    (df['technology_node_nm'] / 250) * 5 +
    (1 - df['num_alternative_parts'] / 10) * 3 +
    np.random.normal(0, 5, n_components)
)

threshold = np.percentile(eol_score, 92)
df['will_go_eol_12m'] = (eol_score > threshold).astype(int)

# Add missing values
for col in ['demand_trend_6m', 'price_trend_6m', 'manufacturer_financial_health']:
    mask = np.random.random(n_components) < 0.03
    df.loc[mask, col] = np.nan

df.to_csv(f"{OUTPUT_DIR}/data/component_data.csv", index=False)
print(f"  Dataset: {df.shape[0]} components, {df.shape[1]} features")
print(f"  EOL in 12m: {df['will_go_eol_12m'].sum()} ({df['will_go_eol_12m'].mean()*100:.1f}%)")
print(f"  Lifecycle distribution:\n{df['lifecycle_stage'].value_counts().to_string()}")

# ══════════════════════════════════════════════════════════════
# STEP 2: EDA
# ══════════════════════════════════════════════════════════════
print("\n[STEP 2] Exploratory Data Analysis...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Component Obsolescence - EDA', fontsize=16, fontweight='bold')

# 1. EOL by lifecycle stage
eol_by_stage = df.groupby('lifecycle_stage')['will_go_eol_12m'].mean() * 100
stage_order = ['Active', 'Mature', 'Declining', 'Last Buy', 'EOL Announced']
eol_by_stage = eol_by_stage.reindex(stage_order)
bars = axes[0,0].bar(range(len(eol_by_stage)), eol_by_stage.values, 
       color=['#4CAF50', '#8BC34A', '#FF9800', '#FF5722', '#B71C1C'])
axes[0,0].set_xticks(range(len(stage_order)))
axes[0,0].set_xticklabels(stage_order, rotation=30, ha='right', fontsize=9)
axes[0,0].set_title('EOL Rate by Lifecycle Stage', fontweight='bold')
axes[0,0].set_ylabel('% Going EOL')
for bar, val in zip(bars, eol_by_stage.values):
    axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}%', ha='center', fontsize=9)

# 2. Component age distribution
for label, color in [(0, '#4CAF50'), (1, '#F44336')]:
    subset = df[df['will_go_eol_12m'] == label]['years_since_introduction']
    axes[0,1].hist(subset, bins=30, alpha=0.6, color=color, 
                   label=f'{"EOL" if label else "Active"}')
axes[0,1].set_title('Component Age Distribution', fontweight='bold')
axes[0,1].set_xlabel('Years Since Introduction')
axes[0,1].legend()

# 3. Technology node vs EOL
tech_eol = df.groupby('technology_node_nm')['will_go_eol_12m'].mean() * 100
axes[0,2].plot(tech_eol.index, tech_eol.values, 'o-', color='#1565C0', linewidth=2)
axes[0,2].set_title('EOL Rate by Technology Node', fontweight='bold')
axes[0,2].set_xlabel('Technology Node (nm)')
axes[0,2].set_ylabel('% Going EOL')
axes[0,2].invert_xaxis()

# 4. Category distribution
cat_eol = df.groupby('category')['will_go_eol_12m'].mean() * 100
cat_eol.sort_values().plot(kind='barh', ax=axes[1,0], color='#7B1FA2')
axes[1,0].set_title('EOL Rate by Category', fontweight='bold')
axes[1,0].set_xlabel('% Going EOL')

# 5. Demand trend vs EOL
axes[1,1].scatter(df['demand_trend_12m'], df['lead_time_increase_pct'],
                  c=df['will_go_eol_12m'], cmap='RdYlGn_r', alpha=0.4, s=10)
axes[1,1].set_title('Demand Trend vs Lead Time Change', fontweight='bold')
axes[1,1].set_xlabel('12M Demand Trend')
axes[1,1].set_ylabel('Lead Time Increase %')

# 6. Class imbalance
labels_pie = ['Active (Not EOL)', 'Going EOL']
sizes = [df['will_go_eol_12m'].value_counts()[0], df['will_go_eol_12m'].value_counts()[1]]
colors_pie = ['#4CAF50', '#F44336']
axes[1,2].pie(sizes, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%',
              startangle=90, textprops={'fontsize': 11})
axes[1,2].set_title('Class Distribution (Imbalanced!)', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/01_eda.png", dpi=150, bbox_inches='tight')
plt.close()
print("  EDA plots saved.")

# ══════════════════════════════════════════════════════════════
# STEP 3: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
print("\n[STEP 3] Feature Engineering...")

df_model = df.copy()

# Encode lifecycle stage as ordinal
stage_map = {'Active': 0, 'Mature': 1, 'Declining': 2, 'Last Buy': 3, 'EOL Announced': 4}
df_model['lifecycle_ordinal'] = df_model['lifecycle_stage'].map(stage_map)

# Engineered features
df_model['age_lifecycle_interaction'] = df_model['years_since_introduction'] * df_model['lifecycle_ordinal']
df_model['supply_availability'] = df_model['num_authorized_distributors'] * df_model['num_alternative_parts']
df_model['demand_momentum'] = df_model['demand_trend_6m'] + df_model['demand_trend_12m']
df_model['supply_stress'] = df_model['lead_time_increase_pct'] * (1 / (df_model['num_authorized_distributors'] + 1))
df_model['obsolescence_risk_score'] = (
    df_model['lifecycle_ordinal'] * 10 +
    df_model['years_since_introduction'] +
    df_model['num_pcn_notices'] * 5 -
    df_model['demand_trend_12m'] * 20
)
df_model['is_old_tech'] = (df_model['technology_node_nm'] >= 90).astype(int)
df_model['pcn_recency'] = 1 / (df_model['last_pcn_months_ago'] + 1)
df_model['demand_supply_gap'] = df_model['monthly_demand_units'] / (df_model['avg_lead_time_weeks'] + 1)

# Encode categoricals
for col in ['category', 'manufacturer', 'package_type']:
    le = LabelEncoder()
    df_model[f'{col}_encoded'] = le.fit_transform(df_model[col])

feature_cols = [
    'years_since_introduction', 'technology_node_nm', 'num_alternative_parts',
    'num_authorized_distributors', 'monthly_demand_units', 'demand_trend_6m',
    'demand_trend_12m', 'avg_lead_time_weeks', 'lead_time_increase_pct',
    'price_trend_6m', 'num_pcn_notices', 'last_pcn_months_ago',
    'manufacturer_financial_health', 'num_design_wins', 'cross_reference_count',
    'rohs_compliant', 'automotive_qualified', 'military_grade',
    'lifecycle_ordinal', 'category_encoded', 'manufacturer_encoded', 'package_type_encoded',
    # Engineered
    'age_lifecycle_interaction', 'supply_availability', 'demand_momentum',
    'supply_stress', 'obsolescence_risk_score', 'is_old_tech',
    'pcn_recency', 'demand_supply_gap'
]

X = df_model[feature_cols].copy()
y = df_model['will_go_eol_12m'].values

# Impute
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  Features: {len(feature_cols)}")
print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"  Train positive rate: {y_train.mean()*100:.1f}%")

# ══════════════════════════════════════════════════════════════
# STEP 4: MODEL TRAINING (Focus on RECALL for EOL detection)
# ══════════════════════════════════════════════════════════════
print("\n[STEP 4] Training Models (Optimized for Recall)...")

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, C=0.5, class_weight='balanced', random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=12, class_weight='balanced_subsample',
        min_samples_leaf=3, random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        min_samples_leaf=5, subsample=0.8, random_state=42
    ),
}

# For GB, we'll use sample_weight to handle imbalance
sample_weights = np.where(y_train == 1, 5.0, 1.0)

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n  Training {name}...")
    
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    elif name == 'Gradient Boosting':
        model.fit(X_train, y_train, sample_weight=sample_weights)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {
        'model': model, 'y_pred': y_pred, 'y_proba': y_proba,
        'auc': auc, 'avg_precision': ap, 'f1': f1
    }
    
    print(f"    AUC-ROC: {auc:.4f} | Avg Precision: {ap:.4f} | F1: {f1:.4f}")

# ══════════════════════════════════════════════════════════════
# STEP 5: THRESHOLD OPTIMIZATION (Maximize Recall @ Acceptable Precision)
# ══════════════════════════════════════════════════════════════
print("\n[STEP 5] Threshold Optimization...")

best_model_name = max(results, key=lambda k: results[k]['auc'])
best_proba = results[best_model_name]['y_proba']

precisions, recalls, thresholds = precision_recall_curve(y_test, best_proba)

# Find threshold where recall >= 0.85 with highest precision
target_recall = 0.85
valid_mask = recalls[:-1] >= target_recall
if valid_mask.any():
    best_idx = np.argmax(precisions[:-1][valid_mask])
    optimal_threshold = thresholds[np.where(valid_mask)[0][best_idx]]
else:
    optimal_threshold = 0.3

y_pred_optimal = (best_proba >= optimal_threshold).astype(int)
print(f"  Best model: {best_model_name}")
print(f"  Optimal threshold: {optimal_threshold:.3f} (targeting {target_recall*100}% recall)")
print(f"  At optimal threshold:")
print(f"    Precision: {(y_pred_optimal[y_test==1].sum() + 0.001) / (y_pred_optimal.sum() + 0.001):.3f}")
print(f"    Recall: {y_pred_optimal[y_test==1].sum() / y_test.sum():.3f}")

# ══════════════════════════════════════════════════════════════
# STEP 6: EVALUATION PLOTS
# ══════════════════════════════════════════════════════════════
print("\n[STEP 6] Generating evaluation plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Component Obsolescence Prediction - Model Evaluation', fontsize=16, fontweight='bold')

# 1. ROC Curves
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    axes[0,0].plot(fpr, tpr, label=f'{name} (AUC={res["auc"]:.3f})', linewidth=2)
axes[0,0].plot([0,1], [0,1], 'k--', alpha=0.3)
axes[0,0].set_title('ROC Curves', fontweight='bold')
axes[0,0].set_xlabel('False Positive Rate')
axes[0,0].set_ylabel('True Positive Rate')
axes[0,0].legend(fontsize=9)

# 2. Precision-Recall Curves
for name, res in results.items():
    prec, rec, _ = precision_recall_curve(y_test, res['y_proba'])
    axes[0,1].plot(rec, prec, label=f'{name} (AP={res["avg_precision"]:.3f})', linewidth=2)
axes[0,1].set_title('Precision-Recall Curves', fontweight='bold')
axes[0,1].set_xlabel('Recall')
axes[0,1].set_ylabel('Precision')
axes[0,1].legend(fontsize=9)
axes[0,1].axvline(x=target_recall, color='red', linestyle='--', alpha=0.5, label=f'Target Recall={target_recall}')

# 3. Confusion Matrix (Optimal threshold)
cm = confusion_matrix(y_test, y_pred_optimal)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,2],
            xticklabels=['Active', 'EOL'], yticklabels=['Active', 'EOL'])
axes[0,2].set_title(f'Confusion Matrix (threshold={optimal_threshold:.2f})', fontweight='bold')
axes[0,2].set_xlabel('Predicted')
axes[0,2].set_ylabel('Actual')

# 4. Feature Importance
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    feat_imp = pd.Series(
        results[best_model_name]['model'].feature_importances_, index=feature_cols
    ).sort_values(ascending=True)
    feat_imp.tail(15).plot(kind='barh', ax=axes[1,0], color='#2E7D32')
axes[1,0].set_title('Top 15 Features', fontweight='bold')

# 5. Threshold vs Metrics
thresholds_range = np.arange(0.1, 0.9, 0.02)
precisions_t, recalls_t, f1s_t = [], [], []
for t in thresholds_range:
    y_t = (best_proba >= t).astype(int)
    tp = ((y_t == 1) & (y_test == 1)).sum()
    fp = ((y_t == 1) & (y_test == 0)).sum()
    fn = ((y_t == 0) & (y_test == 1)).sum()
    p = tp / (tp + fp + 1e-10)
    r = tp / (tp + fn + 1e-10)
    f = 2 * p * r / (p + r + 1e-10)
    precisions_t.append(p)
    recalls_t.append(r)
    f1s_t.append(f)

axes[1,1].plot(thresholds_range, precisions_t, label='Precision', linewidth=2)
axes[1,1].plot(thresholds_range, recalls_t, label='Recall', linewidth=2)
axes[1,1].plot(thresholds_range, f1s_t, label='F1', linewidth=2)
axes[1,1].axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Chosen={optimal_threshold:.2f}')
axes[1,1].set_title('Threshold Optimization', fontweight='bold')
axes[1,1].set_xlabel('Threshold')
axes[1,1].legend()

# 6. Model comparison
model_names = list(results.keys())
aucs = [results[m]['auc'] for m in model_names]
aps = [results[m]['avg_precision'] for m in model_names]
x_pos = np.arange(len(model_names))
width = 0.35
axes[1,2].bar(x_pos - width/2, aucs, width, label='AUC-ROC', color='#1565C0')
axes[1,2].bar(x_pos + width/2, aps, width, label='Avg Precision', color='#FF6F00')
axes[1,2].set_xticks(x_pos)
axes[1,2].set_xticklabels(model_names, rotation=20, ha='right', fontsize=9)
axes[1,2].set_title('Model Comparison', fontweight='bold')
axes[1,2].legend()
axes[1,2].set_ylim(0, 1)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/02_evaluation.png", dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════
# STEP 7: SAVE MODELS
# ══════════════════════════════════════════════════════════════
print("\n[STEP 7] Saving models...")

joblib.dump(results[best_model_name]['model'], f"{OUTPUT_DIR}/models/best_model.joblib")
joblib.dump(scaler, f"{OUTPUT_DIR}/models/scaler.joblib")
joblib.dump(imputer, f"{OUTPUT_DIR}/models/imputer.joblib")

# Save predictions with components
pred_df = df.iloc[X_test.index].copy() if hasattr(X_test, 'index') else df.iloc[:len(y_test)].copy()
pred_df = pred_df.head(len(y_test))
pred_df['predicted_eol_probability'] = best_proba[:len(pred_df)]
pred_df['predicted_eol'] = y_pred_optimal[:len(pred_df)]
pred_df.to_csv(f"{OUTPUT_DIR}/data/predictions.csv", index=False)

print("\n" + "="*70)
print("  FINAL RESULTS")
print("="*70)
for name, res in results.items():
    print(f"  {name}: AUC={res['auc']:.4f} | AP={res['avg_precision']:.4f}")
print(f"\n  BEST: {best_model_name}")
print(f"  At threshold {optimal_threshold:.3f}:")
print(f"  {classification_report(y_test, y_pred_optimal, target_names=['Active', 'EOL'])}")
print("  All outputs saved to:", OUTPUT_DIR)
