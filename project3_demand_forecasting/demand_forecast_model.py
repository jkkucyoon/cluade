"""
╔══════════════════════════════════════════════════════════════════╗
║  PROJECT 3: Component Demand Forecasting System                 ║
║  Author: Ahmed Mohammed Gouda                                   ║
║  Role: ML Engineer (Freelance) - Supply Chain Analytics         ║
║  Date: 2024                                                     ║
║  Client: Global Electronics Component Distributor               ║
╚══════════════════════════════════════════════════════════════════╝

BUSINESS CONTEXT:
    A global distributor needed to forecast demand for top components  
    over the next 3 months to optimize inventory and reduce stockouts.
    They had 3 years of monthly order data for 50 component families.

APPROACH:
    - Time series feature engineering (lags, rolling stats, seasonality)
    - ML approach: treat forecasting as regression with engineered features
    - Compare statistical (Moving Average) vs ML (RF, GB) approaches
    - Evaluate with time-series aware cross-validation

WHY ML OVER TRADITIONAL TIME SERIES:
    - Multiple exogenous features (price, lead time, market indicators)
    - Non-linear relationships
    - Ability to handle multiple time series with shared patterns
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
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
print("  PROJECT 3: COMPONENT DEMAND FORECASTING SYSTEM")
print("="*70)

# ══════════════════════════════════════════════════════════════
# STEP 1: GENERATE REALISTIC TIME SERIES DATA
# ══════════════════════════════════════════════════════════════
print("\n[STEP 1] Generating 3 years of monthly demand data...")

n_components = 50
n_months = 36
dates = pd.date_range('2022-01-01', periods=n_months, freq='MS')

categories = ['Semiconductors', 'Capacitors', 'Resistors', 'Connectors', 'ICs']

all_data = []
for i in range(n_components):
    base_demand = np.random.lognormal(8, 0.8)  # base monthly demand
    trend = np.random.uniform(-0.01, 0.02)  # monthly trend
    seasonality_amp = np.random.uniform(0.05, 0.2)
    noise_level = np.random.uniform(0.05, 0.15)
    
    category = np.random.choice(categories)
    
    for t, date in enumerate(dates):
        # Demand = base * trend * seasonality * noise
        trend_component = 1 + trend * t
        season = 1 + seasonality_amp * np.sin(2 * np.pi * (date.month - 1) / 12)
        # Q4 boost for electronics
        q4_boost = 1.15 if date.month in [10, 11, 12] else 1.0
        # COVID-like disruption in mid-2022
        disruption = 0.7 if date.month in [3,4,5] and date.year == 2022 else 1.0
        # Recovery boost
        recovery = 1.1 if date.month in [7,8,9] and date.year == 2022 else 1.0
        
        noise = 1 + np.random.normal(0, noise_level)
        
        demand = base_demand * trend_component * season * q4_boost * disruption * recovery * noise
        demand = max(100, demand)
        
        # Exogenous features
        avg_price = np.random.lognormal(0, 0.5) * (1 + 0.05 * np.sin(2 * np.pi * t / 12))
        lead_time = max(1, 4 + 2 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 1))
        num_orders = max(1, int(demand / np.random.uniform(50, 200)))
        market_index = 100 + 10 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 3)
        
        all_data.append({
            'component_family': f'COMP-{i:03d}',
            'category': category,
            'date': date,
            'month': date.month,
            'year': date.year,
            'quarter': (date.month - 1) // 3 + 1,
            'demand_units': int(demand),
            'avg_unit_price': round(avg_price, 2),
            'avg_lead_time_weeks': round(lead_time, 1),
            'num_orders': num_orders,
            'market_index': round(market_index, 1),
        })

df = pd.DataFrame(all_data)
df.to_csv(f"{OUTPUT_DIR}/data/demand_data.csv", index=False)

print(f"  Dataset: {df.shape[0]} records ({n_components} components x {n_months} months)")
print(f"  Date range: {dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')}")
print(f"  Avg monthly demand: {df['demand_units'].mean():.0f} units")
print(f"  Categories: {df['category'].nunique()}")

# ══════════════════════════════════════════════════════════════
# STEP 2: EDA
# ══════════════════════════════════════════════════════════════
print("\n[STEP 2] Exploratory Data Analysis...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Demand Forecasting - EDA', fontsize=16, fontweight='bold')

# 1. Overall demand trend
monthly_total = df.groupby('date')['demand_units'].sum()
axes[0,0].plot(monthly_total.index, monthly_total.values, 'b-', linewidth=2)
axes[0,0].fill_between(monthly_total.index, monthly_total.values, alpha=0.2)
axes[0,0].set_title('Total Monthly Demand', fontweight='bold')
axes[0,0].set_ylabel('Total Units')
axes[0,0].tick_params(axis='x', rotation=45)

# 2. Demand by category
for cat in categories:
    cat_demand = df[df['category'] == cat].groupby('date')['demand_units'].sum()
    axes[0,1].plot(cat_demand.index, cat_demand.values, label=cat, linewidth=1.5)
axes[0,1].set_title('Demand by Category', fontweight='bold')
axes[0,1].legend(fontsize=8)
axes[0,1].tick_params(axis='x', rotation=45)

# 3. Seasonality
monthly_avg = df.groupby('month')['demand_units'].mean()
axes[0,2].bar(monthly_avg.index, monthly_avg.values, color='#1565C0')
axes[0,2].set_title('Average Demand by Month (Seasonality)', fontweight='bold')
axes[0,2].set_xlabel('Month')
axes[0,2].set_xticks(range(1, 13))

# 4. Top 10 components
top_comp = df.groupby('component_family')['demand_units'].sum().nlargest(10)
top_comp.sort_values().plot(kind='barh', ax=axes[1,0], color='#7B1FA2')
axes[1,0].set_title('Top 10 Components by Total Demand', fontweight='bold')

# 5. Price vs demand
axes[1,1].scatter(df['avg_unit_price'], df['demand_units'], alpha=0.2, s=5, color='#2E7D32')
axes[1,1].set_title('Price vs Demand', fontweight='bold')
axes[1,1].set_xlabel('Avg Unit Price')
axes[1,1].set_ylabel('Demand Units')
axes[1,1].set_xlim(0, df['avg_unit_price'].quantile(0.95))

# 6. Quarterly trends
quarterly = df.groupby(['year', 'quarter'])['demand_units'].sum().reset_index()
quarterly['period'] = quarterly['year'].astype(str) + '-Q' + quarterly['quarter'].astype(str)
axes[1,2].bar(range(len(quarterly)), quarterly['demand_units'], color='#FF6F00')
axes[1,2].set_xticks(range(len(quarterly)))
axes[1,2].set_xticklabels(quarterly['period'], rotation=45, fontsize=8)
axes[1,2].set_title('Quarterly Demand', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/01_eda.png", dpi=150, bbox_inches='tight')
plt.close()
print("  EDA plots saved.")

# ══════════════════════════════════════════════════════════════
# STEP 3: TIME SERIES FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
print("\n[STEP 3] Time Series Feature Engineering...")

df_sorted = df.sort_values(['component_family', 'date']).reset_index(drop=True)

# Create lag features and rolling stats per component
def create_ts_features(group):
    g = group.copy()
    # Lag features
    for lag in [1, 2, 3, 6]:
        g[f'demand_lag_{lag}'] = g['demand_units'].shift(lag)
    
    # Rolling statistics
    for window in [3, 6]:
        g[f'demand_rolling_mean_{window}'] = g['demand_units'].shift(1).rolling(window).mean()
        g[f'demand_rolling_std_{window}'] = g['demand_units'].shift(1).rolling(window).std()
    
    # Rolling min/max
    g['demand_rolling_min_3'] = g['demand_units'].shift(1).rolling(3).min()
    g['demand_rolling_max_3'] = g['demand_units'].shift(1).rolling(3).max()
    
    # Percent change
    g['demand_pct_change'] = g['demand_units'].pct_change()
    
    # Exponential moving average
    g['demand_ema_3'] = g['demand_units'].shift(1).ewm(span=3).mean()
    
    # Price and lead time lags
    g['price_lag_1'] = g['avg_unit_price'].shift(1)
    g['lead_time_lag_1'] = g['avg_lead_time_weeks'].shift(1)
    
    return g

df_feat = df_sorted.groupby('component_family', group_keys=False).apply(create_ts_features)

# Encode category
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_feat['category_encoded'] = le.fit_transform(df_feat['category'])

# Drop rows with NaN from lag/rolling (first 6 months per component)
df_feat = df_feat.dropna(subset=['demand_lag_6', 'demand_rolling_mean_6']).reset_index(drop=True)

feature_cols = [
    'month', 'quarter', 'category_encoded',
    'demand_lag_1', 'demand_lag_2', 'demand_lag_3', 'demand_lag_6',
    'demand_rolling_mean_3', 'demand_rolling_mean_6',
    'demand_rolling_std_3', 'demand_rolling_std_6',
    'demand_rolling_min_3', 'demand_rolling_max_3',
    'demand_pct_change', 'demand_ema_3',
    'avg_unit_price', 'avg_lead_time_weeks', 'num_orders', 'market_index',
    'price_lag_1', 'lead_time_lag_1'
]

print(f"  Created {len(feature_cols)} features")
print(f"  Dataset after feature eng: {df_feat.shape[0]} records")

# ══════════════════════════════════════════════════════════════
# STEP 4: TIME-SERIES AWARE TRAIN/TEST SPLIT
# ══════════════════════════════════════════════════════════════
print("\n[STEP 4] Time-Series Train/Test Split...")

# Use last 3 months as test (most recent data)
cutoff_date = dates[-3]
train_mask = df_feat['date'] < cutoff_date
test_mask = df_feat['date'] >= cutoff_date

X_train = df_feat.loc[train_mask, feature_cols]
y_train = df_feat.loc[train_mask, 'demand_units']
X_test = df_feat.loc[test_mask, feature_cols]
y_test = df_feat.loc[test_mask, 'demand_units']

print(f"  Train: {X_train.shape[0]} records (up to {cutoff_date.strftime('%Y-%m')})")
print(f"  Test: {X_test.shape[0]} records (last 3 months)")

# ══════════════════════════════════════════════════════════════
# STEP 5: MODEL TRAINING
# ══════════════════════════════════════════════════════════════
print("\n[STEP 5] Training Models...")

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=10.0),
    'Random Forest': RandomForestRegressor(
        n_estimators=100, max_depth=15, min_samples_leaf=3, random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        min_samples_leaf=5, subsample=0.8, random_state=42
    ),
}

# Simple baselines
naive_pred = df_feat.loc[test_mask, 'demand_lag_1'].values  # Last month as prediction
ma3_pred = df_feat.loc[test_mask, 'demand_rolling_mean_3'].values  # 3-month moving average

print(f"\n  Baseline (Naive - Last Month): MAE={mean_absolute_error(y_test, naive_pred):.0f} | "
      f"RMSE={np.sqrt(mean_squared_error(y_test, naive_pred)):.0f} | R2={r2_score(y_test, naive_pred):.4f}")
print(f"  Baseline (3M Moving Avg): MAE={mean_absolute_error(y_test, ma3_pred):.0f} | "
      f"RMSE={np.sqrt(mean_squared_error(y_test, ma3_pred)):.0f} | R2={r2_score(y_test, ma3_pred):.4f}")

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = np.maximum(y_pred, 0)  # Demand can't be negative
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100
    
    results[name] = {
        'model': model, 'y_pred': y_pred,
        'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape
    }
    
    print(f"  {name}: MAE={mae:.0f} | RMSE={rmse:.0f} | R2={r2:.4f} | MAPE={mape:.1f}%")

# ══════════════════════════════════════════════════════════════
# STEP 6: TIME SERIES CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════
print("\n[STEP 6] Time Series Cross-Validation (Expanding Window)...")

unique_dates = sorted(df_feat['date'].unique())
n_dates = len(unique_dates)

# Use last 6 months for validation windows
cv_results = {name: [] for name in models.keys()}

for fold in range(3):
    test_start = n_dates - 3 - fold * 3
    test_end = test_start + 3
    
    train_dates = unique_dates[:test_start]
    test_dates = unique_dates[test_start:test_end]
    
    cv_train = df_feat[df_feat['date'].isin(train_dates)]
    cv_test = df_feat[df_feat['date'].isin(test_dates)]
    
    for name, model_template in models.items():
        from sklearn.base import clone
        m = clone(model_template)
        m.fit(cv_train[feature_cols], cv_train['demand_units'])
        pred = np.maximum(m.predict(cv_test[feature_cols]), 0)
        mae = mean_absolute_error(cv_test['demand_units'], pred)
        cv_results[name].append(mae)

print("  Time Series CV Results (MAE):")
for name, scores in cv_results.items():
    print(f"    {name}: {np.mean(scores):.0f} +/- {np.std(scores):.0f}")

# ══════════════════════════════════════════════════════════════
# STEP 7: COMPREHENSIVE PLOTS
# ══════════════════════════════════════════════════════════════
print("\n[STEP 7] Generating evaluation plots...")

best_model_name = min(results, key=lambda k: results[k]['mae'])
best_pred = results[best_model_name]['y_pred']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Demand Forecasting - Model Evaluation', fontsize=16, fontweight='bold')

# 1. Actual vs Predicted (scatter)
axes[0,0].scatter(y_test, best_pred, alpha=0.3, s=15, color='#1565C0')
max_val = max(y_test.max(), best_pred.max())
axes[0,0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect')
axes[0,0].set_title(f'Actual vs Predicted ({best_model_name})', fontweight='bold')
axes[0,0].set_xlabel('Actual Demand')
axes[0,0].set_ylabel('Predicted Demand')
axes[0,0].legend()

# 2. Model comparison
model_names = ['Naive', '3M MA'] + list(results.keys())
maes = [
    mean_absolute_error(y_test, naive_pred),
    mean_absolute_error(y_test, ma3_pred)
] + [results[m]['mae'] for m in results.keys()]
colors = ['#999999', '#999999'] + ['#1565C0'] * len(results)
bars = axes[0,1].bar(range(len(model_names)), maes, color=colors)
axes[0,1].set_xticks(range(len(model_names)))
axes[0,1].set_xticklabels(model_names, rotation=30, ha='right', fontsize=8)
axes[0,1].set_title('MAE Comparison (Lower = Better)', fontweight='bold')
axes[0,1].set_ylabel('MAE')
for bar, val in zip(bars, maes):
    axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                   f'{val:.0f}', ha='center', fontsize=8)

# 3. Residual distribution
residuals = y_test.values - best_pred
axes[0,2].hist(residuals, bins=50, color='#7B1FA2', alpha=0.7, edgecolor='white')
axes[0,2].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0,2].set_title('Residual Distribution', fontweight='bold')
axes[0,2].set_xlabel('Residual (Actual - Predicted)')

# 4. Feature importance
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    feat_imp = pd.Series(
        results[best_model_name]['model'].feature_importances_, index=feature_cols
    ).sort_values(ascending=True)
    feat_imp.tail(12).plot(kind='barh', ax=axes[1,0], color='#2E7D32')
axes[1,0].set_title('Top Feature Importance', fontweight='bold')

# 5. Time series of predictions for top component
top_comp = df.groupby('component_family')['demand_units'].sum().idxmax()
comp_data = df_feat[df_feat['component_family'] == top_comp].sort_values('date') if 'component_family' in df_feat.columns else None

if comp_data is not None and len(comp_data) > 0:
    comp_test = comp_data[comp_data['date'] >= cutoff_date]
    comp_train = comp_data[comp_data['date'] < cutoff_date]
    comp_pred = results[best_model_name]['model'].predict(comp_test[feature_cols])
    axes[1,1].plot(comp_train['date'], comp_train['demand_units'], 'b-', label='Historical', linewidth=2)
    axes[1,1].plot(comp_test['date'], comp_test['demand_units'], 'g-o', label='Actual', linewidth=2)
    axes[1,1].plot(comp_test['date'], comp_pred, 'r--o', label='Predicted', linewidth=2)
    axes[1,1].set_title(f'Forecast for {top_comp}', fontweight='bold')
    axes[1,1].legend()
    axes[1,1].tick_params(axis='x', rotation=45)

# 6. MAPE by category
test_with_pred = df_feat.loc[test_mask].copy()
test_with_pred['prediction'] = best_pred
cat_mape = test_with_pred.groupby('category').apply(
    lambda g: np.mean(np.abs((g['demand_units'] - g['prediction']) / (g['demand_units'] + 1))) * 100
).sort_values()
cat_mape.plot(kind='barh', ax=axes[1,2], color='#FF6F00')
axes[1,2].set_title('MAPE by Category', fontweight='bold')
axes[1,2].set_xlabel('MAPE (%)')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/02_evaluation.png", dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════
# STEP 8: SAVE MODELS & OUTPUTS
# ══════════════════════════════════════════════════════════════
print("\n[STEP 8] Saving models and outputs...")

joblib.dump(results[best_model_name]['model'], f"{OUTPUT_DIR}/models/best_forecasting_model.joblib")

summary = pd.DataFrame({
    'Model': list(results.keys()),
    'MAE': [results[m]['mae'] for m in results],
    'RMSE': [results[m]['rmse'] for m in results],
    'R2': [results[m]['r2'] for m in results],
    'MAPE%': [results[m]['mape'] for m in results],
}).round(2)
summary.to_csv(f"{OUTPUT_DIR}/data/model_comparison.csv", index=False)

print("\n" + "="*70)
print("  FINAL RESULTS")
print("="*70)
print(summary.to_string(index=False))
print(f"\n  Best Model: {best_model_name}")
print(f"  MAE: {results[best_model_name]['mae']:.0f} units")
print(f"  MAPE: {results[best_model_name]['mape']:.1f}%")
print(f"  R2: {results[best_model_name]['r2']:.4f}")
print("  All outputs saved to:", OUTPUT_DIR)
