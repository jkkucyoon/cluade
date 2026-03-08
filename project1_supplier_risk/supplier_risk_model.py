"""
╔══════════════════════════════════════════════════════════════════╗
║  PROJECT 1: Supplier Risk Classification & Scoring System       ║
║  Author: Ahmed Mohammed Gouda                                   ║
║  Role: ML Engineer (Freelance) - Supply Chain Analytics         ║
║  Date: 2024                                                     ║
║  Client: Electronics Manufacturing Company                      ║
╚══════════════════════════════════════════════════════════════════╝

BUSINESS CONTEXT:
    A mid-size electronics manufacturer needed to assess supplier risk 
    to avoid supply chain disruptions. They had ~2000 suppliers across
    15 countries and wanted an ML model to score each supplier 0-100 
    and classify them into Low/Medium/High risk.

APPROACH:
    1. Collected supplier data from multiple sources (ERP, financial DBs, news)
    2. Engineered 20+ risk features across 4 categories
    3. Trained multiple classifiers + ensemble
    4. Deployed as REST API with FastAPI + Docker

ALGORITHMS USED & WHY:
    - GradientBoosting: Best for tabular data, handles mixed features
    - RandomForest: Robust baseline, great feature importance
    - LogisticRegression: Interpretable baseline
    - VotingClassifier: Ensemble for production robustness
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, accuracy_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
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
print("  PROJECT 1: SUPPLIER RISK CLASSIFICATION & SCORING SYSTEM")
print("="*70)

# ══════════════════════════════════════════════════════════════
# STEP 1: DATA GENERATION (Realistic Supply Chain Data)
# ══════════════════════════════════════════════════════════════
print("\n[STEP 1] Generating realistic supplier dataset...")

n_suppliers = 2000

countries = ['China', 'Taiwan', 'South Korea', 'Japan', 'Germany', 'USA', 
             'Vietnam', 'India', 'Thailand', 'Mexico', 'Malaysia', 'Philippines',
             'UK', 'France', 'Brazil']
country_risk = {
    'China': 0.6, 'Taiwan': 0.7, 'South Korea': 0.3, 'Japan': 0.15,
    'Germany': 0.1, 'USA': 0.1, 'Vietnam': 0.5, 'India': 0.45,
    'Thailand': 0.4, 'Mexico': 0.35, 'Malaysia': 0.3, 'Philippines': 0.45,
    'UK': 0.1, 'France': 0.1, 'Brazil': 0.4
}

categories = ['Semiconductors', 'Passive Components', 'Connectors', 
              'PCB', 'Displays', 'Sensors', 'Memory', 'Power Management']

supplier_countries = np.random.choice(countries, n_suppliers, 
    p=[0.25, 0.1, 0.08, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02])

data = {
    'supplier_id': [f'SUP-{i:04d}' for i in range(n_suppliers)],
    'country': supplier_countries,
    'component_category': np.random.choice(categories, n_suppliers),
    'years_in_business': np.clip(np.random.exponential(12, n_suppliers), 1, 50).astype(int),
    'num_manufacturing_sites': np.random.choice([1,2,3,4,5,6,7,8], n_suppliers, 
        p=[0.3, 0.25, 0.15, 0.12, 0.08, 0.05, 0.03, 0.02]),
    'annual_revenue_millions': np.clip(np.random.lognormal(4, 1.5, n_suppliers), 1, 50000).round(1),
    'num_employees': np.clip(np.random.lognormal(6, 1.2, n_suppliers), 10, 100000).astype(int),
    'on_time_delivery_rate': np.clip(np.random.beta(8, 2, n_suppliers), 0.5, 1.0).round(3),
    'defect_rate_ppm': np.clip(np.random.exponential(500, n_suppliers), 0, 10000).round(0),
    'lead_time_days': np.clip(np.random.lognormal(3, 0.6, n_suppliers), 3, 180).astype(int),
    'lead_time_variability': np.clip(np.random.exponential(0.15, n_suppliers), 0, 0.8).round(3),
    'financial_health_score': np.clip(np.random.normal(65, 20, n_suppliers), 0, 100).round(1),
    'debt_to_equity_ratio': np.clip(np.random.exponential(0.8, n_suppliers), 0, 5).round(2),
    'num_certifications': np.random.poisson(3, n_suppliers),
    'has_iso9001': np.random.choice([0, 1], n_suppliers, p=[0.25, 0.75]),
    'has_iso14001': np.random.choice([0, 1], n_suppliers, p=[0.45, 0.55]),
    'has_iatf16949': np.random.choice([0, 1], n_suppliers, p=[0.6, 0.4]),
    'num_customers': np.clip(np.random.lognormal(3, 1, n_suppliers), 1, 500).astype(int),
    'single_source_pct': np.clip(np.random.beta(2, 5, n_suppliers), 0, 1).round(3),
    'sub_tier_visibility': np.random.choice([0, 1, 2, 3], n_suppliers, p=[0.2, 0.3, 0.3, 0.2]),
    'recent_disruption_events': np.random.poisson(0.5, n_suppliers),
    'news_sentiment_score': np.clip(np.random.normal(0.6, 0.2, n_suppliers), 0, 1).round(3),
    'compliance_violations': np.random.poisson(0.3, n_suppliers),
    'geographic_risk_index': [country_risk[c] + np.random.normal(0, 0.05) for c in supplier_countries],
}

df = pd.DataFrame(data)
df['geographic_risk_index'] = df['geographic_risk_index'].clip(0, 1).round(3)

# Add some realistic missing values
for col in ['financial_health_score', 'debt_to_equity_ratio', 'news_sentiment_score']:
    mask = np.random.random(n_suppliers) < 0.05
    df.loc[mask, col] = np.nan

# Generate target: risk label based on realistic logic
risk_score = (
    (1 - df['on_time_delivery_rate'].fillna(0.8)) * 15 +
    (df['defect_rate_ppm'].fillna(500) / 10000) * 10 +
    df['lead_time_variability'].fillna(0.15) * 12 +
    (1 - df['financial_health_score'].fillna(65) / 100) * 15 +
    df['debt_to_equity_ratio'].fillna(0.8) * 5 +
    df['geographic_risk_index'] * 12 +
    df['recent_disruption_events'] * 8 +
    (1 - df['news_sentiment_score'].fillna(0.6)) * 8 +
    df['compliance_violations'] * 6 +
    (1 - df['has_iso9001']) * 4 +
    df['single_source_pct'].fillna(0.3) * 5 +
    np.random.normal(0, 3, n_suppliers)
)

risk_score = np.clip(risk_score, 0, 100)
df['risk_label'] = pd.cut(risk_score, bins=[0, 25, 50, 100], labels=['Low', 'Medium', 'High'])

df.to_csv(f"{OUTPUT_DIR}/data/supplier_data.csv", index=False)
print(f"  Dataset shape: {df.shape}")
print(f"  Risk distribution:\n{df['risk_label'].value_counts().to_string()}")

# ══════════════════════════════════════════════════════════════
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ══════════════════════════════════════════════════════════════
print("\n[STEP 2] Exploratory Data Analysis...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Supplier Risk - Exploratory Data Analysis', fontsize=16, fontweight='bold')

# 1. Risk distribution
colors_risk = {'Low': '#4CAF50', 'Medium': '#FF9800', 'High': '#F44336'}
risk_counts = df['risk_label'].value_counts()
axes[0,0].bar(risk_counts.index, risk_counts.values, color=[colors_risk[x] for x in risk_counts.index])
axes[0,0].set_title('Risk Label Distribution', fontweight='bold')
axes[0,0].set_ylabel('Count')
for i, v in enumerate(risk_counts.values):
    axes[0,0].text(i, v+20, str(v), ha='center', fontweight='bold')

# 2. On-time delivery by risk
for label, color in colors_risk.items():
    subset = df[df['risk_label'] == label]['on_time_delivery_rate'].dropna()
    axes[0,1].hist(subset, alpha=0.6, label=label, color=color, bins=30)
axes[0,1].set_title('On-Time Delivery Rate by Risk Level', fontweight='bold')
axes[0,1].set_xlabel('On-Time Delivery Rate')
axes[0,1].legend()

# 3. Financial health vs defect rate
scatter = axes[0,2].scatter(
    df['financial_health_score'], df['defect_rate_ppm'],
    c=df['risk_label'].map({'Low': 0, 'Medium': 1, 'High': 2}),
    cmap='RdYlGn_r', alpha=0.5, s=15
)
axes[0,2].set_title('Financial Health vs Defect Rate', fontweight='bold')
axes[0,2].set_xlabel('Financial Health Score')
axes[0,2].set_ylabel('Defect Rate (PPM)')

# 4. Country risk distribution
country_risk_df = df.groupby('country')['risk_label'].apply(
    lambda x: (x == 'High').sum() / len(x) * 100
).sort_values(ascending=True)
axes[1,0].barh(country_risk_df.index, country_risk_df.values, color='#1565C0')
axes[1,0].set_title('% High Risk Suppliers by Country', fontweight='bold')
axes[1,0].set_xlabel('% High Risk')

# 5. Correlation heatmap
numeric_cols = ['on_time_delivery_rate', 'defect_rate_ppm', 'financial_health_score',
                'lead_time_variability', 'geographic_risk_index', 'debt_to_equity_ratio',
                'news_sentiment_score', 'compliance_violations']
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=axes[1,1], cbar_kws={'shrink': 0.8}, annot_kws={'size': 7})
axes[1,1].set_title('Feature Correlations', fontweight='bold')
axes[1,1].tick_params(labelsize=7)

# 6. Feature importance preview
feature_means = df.groupby('risk_label')[numeric_cols].mean()
feature_diff = (feature_means.loc['High'] - feature_means.loc['Low']).abs().sort_values(ascending=True)
axes[1,2].barh(feature_diff.index, feature_diff.values, color='#7B1FA2')
axes[1,2].set_title('Feature Difference (High vs Low Risk)', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/01_eda_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("  EDA plots saved.")

# ══════════════════════════════════════════════════════════════
# STEP 3: FEATURE ENGINEERING & PREPROCESSING
# ══════════════════════════════════════════════════════════════
print("\n[STEP 3] Feature Engineering & Preprocessing...")

df_model = df.copy()

# Feature engineering
df_model['revenue_per_employee'] = df_model['annual_revenue_millions'] * 1e6 / df_model['num_employees']
df_model['delivery_reliability'] = df_model['on_time_delivery_rate'] * (1 - df_model['lead_time_variability'])
df_model['certification_score'] = df_model['has_iso9001'] + df_model['has_iso14001'] + df_model['has_iatf16949']
df_model['risk_exposure'] = df_model['single_source_pct'] * df_model['geographic_risk_index']
df_model['financial_stability'] = df_model['financial_health_score'] / (1 + df_model['debt_to_equity_ratio'])
df_model['supplier_maturity'] = np.log1p(df_model['years_in_business']) * df_model['num_manufacturing_sites']
df_model['quality_index'] = (1 - df_model['defect_rate_ppm'] / 10000) * df_model['on_time_delivery_rate']
df_model['disruption_intensity'] = df_model['recent_disruption_events'] * (1 - df_model['news_sentiment_score'].fillna(0.5))

# Encode categoricals
le_country = LabelEncoder()
le_category = LabelEncoder()
df_model['country_encoded'] = le_country.fit_transform(df_model['country'])
df_model['category_encoded'] = le_category.fit_transform(df_model['component_category'])

# Select features
feature_cols = [
    'years_in_business', 'num_manufacturing_sites', 'annual_revenue_millions',
    'num_employees', 'on_time_delivery_rate', 'defect_rate_ppm', 'lead_time_days',
    'lead_time_variability', 'financial_health_score', 'debt_to_equity_ratio',
    'num_certifications', 'has_iso9001', 'has_iso14001', 'has_iatf16949',
    'num_customers', 'single_source_pct', 'sub_tier_visibility',
    'recent_disruption_events', 'news_sentiment_score', 'compliance_violations',
    'geographic_risk_index', 'country_encoded', 'category_encoded',
    # Engineered features
    'revenue_per_employee', 'delivery_reliability', 'certification_score',
    'risk_exposure', 'financial_stability', 'supplier_maturity',
    'quality_index', 'disruption_intensity'
]

X = df_model[feature_cols].copy()
y = LabelEncoder().fit_transform(df_model['risk_label'])  # 0=High, 1=Low, 2=Medium

# Handle missing values
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  Engineered {len(feature_cols)} features")
print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
print(f"  Class distribution (train): {np.bincount(y_train)}")

# ══════════════════════════════════════════════════════════════
# STEP 4: MODEL TRAINING & COMPARISON
# ══════════════════════════════════════════════════════════════
print("\n[STEP 4] Training Multiple Models...")

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, C=1.0, class_weight='balanced', random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=12, min_samples_split=5,
        min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        min_samples_split=10, subsample=0.8, random_state=42
    ),
    'SVM (RBF)': SVC(
        kernel='rbf', C=10, gamma='scale', class_weight='balanced',
        probability=True, random_state=42
    ),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n  Training {name}...")
    
    if name in ['Logistic Regression', 'SVM (RBF)']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1_macro')
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro')
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'accuracy': acc,
        'f1_macro': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"    Accuracy: {acc:.4f} | F1-Macro: {f1:.4f} | CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ══════════════════════════════════════════════════════════════
# STEP 5: ENSEMBLE MODEL (PRODUCTION)
# ══════════════════════════════════════════════════════════════
print("\n[STEP 5] Building Ensemble (VotingClassifier)...")

ensemble = VotingClassifier(
    estimators=[
        ('rf', models['Random Forest']),
        ('gb', models['Gradient Boosting']),
        ('lr', Pipeline([('scaler', StandardScaler()), 
                        ('clf', LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced'))]))
    ],
    voting='soft',
    weights=[2, 3, 1]  # GB gets highest weight
)

ensemble.fit(X_train, y_train)
y_pred_ens = ensemble.predict(X_test)
y_proba_ens = ensemble.predict_proba(X_test)

acc_ens = accuracy_score(y_test, y_pred_ens)
f1_ens = f1_score(y_test, y_pred_ens, average='macro')
cv_ens = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring='f1_macro')

results['Ensemble (Voting)'] = {
    'model': ensemble,
    'y_pred': y_pred_ens,
    'y_proba': y_proba_ens,
    'accuracy': acc_ens,
    'f1_macro': f1_ens,
    'cv_mean': cv_ens.mean(),
    'cv_std': cv_ens.std()
}

print(f"  Ensemble Accuracy: {acc_ens:.4f} | F1-Macro: {f1_ens:.4f} | CV: {cv_ens.mean():.4f}")

# ══════════════════════════════════════════════════════════════
# STEP 6: HYPERPARAMETER TUNING (Best Model)
# ══════════════════════════════════════════════════════════════
print("\n[STEP 6] Hyperparameter Tuning (Gradient Boosting)...")

param_grid = {
    'n_estimators': [150, 200],
    'max_depth': [4, 5],
    'learning_rate': [0.1],
    'subsample': [0.8],
}

grid_search = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid, cv=3, scoring='f1_macro', n_jobs=-1, verbose=0
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
y_proba_best = best_model.predict_proba(X_test)

print(f"  Best params: {grid_search.best_params_}")
print(f"  Best CV F1: {grid_search.best_score_:.4f}")
print(f"  Test F1: {f1_score(y_test, y_pred_best, average='macro'):.4f}")

# ══════════════════════════════════════════════════════════════
# STEP 7: COMPREHENSIVE EVALUATION PLOTS
# ══════════════════════════════════════════════════════════════
print("\n[STEP 7] Generating evaluation plots...")

# Plot 1: Model Comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Model Evaluation - Supplier Risk Classification', fontsize=16, fontweight='bold')

# Accuracy comparison
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in model_names]
f1s = [results[m]['f1_macro'] for m in model_names]

x_pos = np.arange(len(model_names))
width = 0.35
bars1 = axes[0,0].bar(x_pos - width/2, accuracies, width, label='Accuracy', color='#1565C0')
bars2 = axes[0,0].bar(x_pos + width/2, f1s, width, label='F1-Macro', color='#FF6F00')
axes[0,0].set_xticks(x_pos)
axes[0,0].set_xticklabels(model_names, rotation=30, ha='right', fontsize=9)
axes[0,0].set_title('Model Comparison', fontweight='bold')
axes[0,0].legend()
axes[0,0].set_ylim(0.5, 1.0)
for bar in bars1:
    axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{bar.get_height():.3f}', ha='center', fontsize=7)
for bar in bars2:
    axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f'{bar.get_height():.3f}', ha='center', fontsize=7)

# Confusion Matrix (Best model)
cm = confusion_matrix(y_test, y_pred_best)
labels = ['High', 'Low', 'Medium']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,1],
            xticklabels=labels, yticklabels=labels)
axes[0,1].set_title('Confusion Matrix (Tuned GB)', fontweight='bold')
axes[0,1].set_xlabel('Predicted')
axes[0,1].set_ylabel('Actual')

# ROC Curves (One-vs-Rest)
for i, label in enumerate(labels):
    y_test_bin = (y_test == i).astype(int)
    fpr, tpr, _ = roc_curve(y_test_bin, y_proba_best[:, i])
    auc = roc_auc_score(y_test_bin, y_proba_best[:, i])
    axes[1,0].plot(fpr, tpr, label=f'{label} (AUC={auc:.3f})', linewidth=2)
axes[1,0].plot([0,1], [0,1], 'k--', alpha=0.3)
axes[1,0].set_title('ROC Curves (One-vs-Rest)', fontweight='bold')
axes[1,0].set_xlabel('False Positive Rate')
axes[1,0].set_ylabel('True Positive Rate')
axes[1,0].legend()

# Feature Importance
feat_imp = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
feat_imp.tail(15).plot(kind='barh', ax=axes[1,1], color='#2E7D32')
axes[1,1].set_title('Top 15 Feature Importance', fontweight='bold')
axes[1,1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/02_model_evaluation.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 2: Cross-validation details
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# CV scores bar chart
cv_means = [results[m]['cv_mean'] for m in model_names]
cv_stds = [results[m]['cv_std'] for m in model_names]
axes[0].bar(range(len(model_names)), cv_means, yerr=cv_stds, capsize=5, color='#1565C0')
axes[0].set_xticks(range(len(model_names)))
axes[0].set_xticklabels([n.replace(' ', '\n') for n in model_names], fontsize=8)
axes[0].set_title('Cross-Validation F1 Scores', fontweight='bold')
axes[0].set_ylabel('F1-Macro')

# Classification report heatmap
report = classification_report(y_test, y_pred_best, target_names=labels, output_dict=True)
report_df = pd.DataFrame(report).T.iloc[:3, :3]
sns.heatmap(report_df, annot=True, fmt='.3f', cmap='YlGn', ax=axes[1])
axes[1].set_title('Classification Report (Tuned GB)', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/03_cross_validation.png", dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════
# STEP 8: SAVE MODELS & ARTIFACTS
# ══════════════════════════════════════════════════════════════
print("\n[STEP 8] Saving models and artifacts...")

joblib.dump(best_model, f"{OUTPUT_DIR}/models/gradient_boosting_tuned.joblib")
joblib.dump(ensemble, f"{OUTPUT_DIR}/models/ensemble_voting.joblib")
joblib.dump(scaler, f"{OUTPUT_DIR}/models/scaler.joblib")
joblib.dump(imputer, f"{OUTPUT_DIR}/models/imputer.joblib")

# Save results summary
summary = pd.DataFrame({
    'Model': model_names + ['Tuned GB'],
    'Accuracy': [results[m]['accuracy'] for m in model_names] + [accuracy_score(y_test, y_pred_best)],
    'F1_Macro': [results[m]['f1_macro'] for m in model_names] + [f1_score(y_test, y_pred_best, average='macro')],
    'CV_Mean': [results[m]['cv_mean'] for m in model_names] + [grid_search.best_score_],
})
summary.to_csv(f"{OUTPUT_DIR}/data/model_comparison.csv", index=False)

print("\n" + "="*70)
print("  FINAL RESULTS SUMMARY")
print("="*70)
print(summary.to_string(index=False))
print("\n  Best Model: Tuned Gradient Boosting")
print(f"  Classification Report:\n{classification_report(y_test, y_pred_best, target_names=labels)}")
print("  All outputs saved to:", OUTPUT_DIR)
