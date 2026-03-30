---
name: data-analysis
description: Comprehensive data analysis skill with statistical methods, visualization patterns, and ML pipeline templates.
---

# Data Analysis Skill

## Overview

This skill provides a comprehensive framework for data analysis tasks, including:
- Statistical analysis and hypothesis testing
- Data visualization patterns and best practices
- Machine learning pipeline templates
- Data cleaning and preprocessing workflows

## Statistical Analysis Methods

### Descriptive Statistics
When analyzing data, always start with descriptive statistics:
1. Central tendency: mean, median, mode
2. Dispersion: variance, standard deviation, IQR
3. Shape: skewness, kurtosis
4. Missing data patterns: MCAR, MAR, MNAR

### Hypothesis Testing Framework
Follow these steps for any hypothesis test:
1. State null hypothesis (H0) and alternative hypothesis (H1)
2. Choose significance level (alpha = 0.05 by default)
3. Select appropriate test:
   - t-test: comparing means of two groups
   - ANOVA: comparing means of 3+ groups
   - Chi-square: testing independence of categorical variables
   - Mann-Whitney U: non-parametric alternative to t-test
   - Kruskal-Wallis: non-parametric alternative to ANOVA
4. Calculate test statistic and p-value
5. Make decision and report effect size

### Regression Analysis
For regression tasks:
- Linear regression: continuous outcome, linear relationship
- Logistic regression: binary outcome
- Polynomial regression: non-linear relationships
- Ridge/Lasso: when multicollinearity or feature selection needed
- Random Forest/XGBoost: complex non-linear patterns

Always report:
- R-squared and adjusted R-squared
- RMSE and MAE for continuous outcomes
- AUC-ROC and confusion matrix for classification
- Residual plots for model diagnostics

## Data Visualization Patterns

### Chart Selection Guide
| Data Type | Comparison | Distribution | Relationship | Composition |
|-----------|-----------|--------------|-------------|-------------|
| Categorical | Bar chart | — | — | Pie/Donut |
| Continuous | Box plot | Histogram/KDE | Scatter plot | Stacked area |
| Time series | Line chart | — | — | Stacked area |
| Geographic | Choropleth | — | — | — |
| Hierarchical | Treemap | — | Sankey | Sunburst |

### Visualization Best Practices
1. Always label axes with units
2. Use colorblind-friendly palettes (viridis, cividis)
3. Start y-axis at zero for bar charts
4. Use log scale when data spans multiple orders of magnitude
5. Include confidence intervals when showing estimates
6. Avoid 3D charts — they distort perception
7. Keep data-ink ratio high (minimize chartjunk)

### Python Visualization Code Templates

#### Matplotlib Basic Template
```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Analysis Dashboard', fontsize=16, fontweight='bold')

# Distribution
axes[0, 0].hist(data, bins=30, edgecolor='white', alpha=0.7)
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution')

# Time series
axes[0, 1].plot(dates, values, linewidth=1.5)
axes[0, 1].fill_between(dates, lower, upper, alpha=0.2)
axes[0, 1].set_title('Trend Over Time')

# Correlation
scatter = axes[1, 0].scatter(x, y, c=z, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, ax=axes[1, 0])
axes[1, 0].set_title('Correlation')

# Categories
axes[1, 1].barh(categories, values, color='steelblue')
axes[1, 1].set_title('Category Comparison')

plt.tight_layout()
plt.savefig('dashboard.png', dpi=150, bbox_inches='tight')
```

#### Seaborn Statistical Plots
```python
import seaborn as sns

# Pair plot for multi-variable exploration
g = sns.pairplot(df, hue='category', diag_kind='kde')
g.fig.suptitle('Pairwise Relationships', y=1.02)

# Heatmap for correlation matrix
plt.figure(figsize=(10, 8))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1)

# Violin plot for distributions
sns.violinplot(data=df, x='group', y='value', inner='box')
```

## Machine Learning Pipeline Templates

### Classification Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(random_state=42))
])

# Hyperparameter tuning
param_grid = {
    'clf__n_estimators': [100, 200, 500],
    'clf__max_depth': [5, 10, 20, None],
    'clf__min_samples_split': [2, 5, 10],
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)

# Evaluation
y_pred = grid.predict(X_test)
y_prob = grid.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print(f'AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}')
```

### Time Series Forecasting
```python
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

# Decomposition
decomposition = seasonal_decompose(ts, model='multiplicative', period=12)
fig = decomposition.plot()

# SARIMA
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit(disp=False)

# Forecast
forecast = results.forecast(steps=24)
conf_int = results.get_forecast(steps=24).conf_int()
```

### Feature Engineering Checklist
- [ ] Handle missing values (imputation strategy documented)
- [ ] Encode categorical variables (one-hot, target encoding, ordinal)
- [ ] Scale numerical features (standardize for linear models, normalize for distance-based)
- [ ] Create interaction features where domain knowledge suggests
- [ ] Extract date/time features (day of week, month, hour, is_weekend)
- [ ] Apply log transform to skewed distributions
- [ ] Generate polynomial features for suspected non-linear relationships
- [ ] Calculate rolling statistics for time series features
- [ ] Create lag features for temporal dependencies
- [ ] Bin continuous variables when non-linear relationships exist

## Data Cleaning Workflow

### Step 1: Initial Assessment
```python
def data_quality_report(df):
    report = pd.DataFrame({
        'dtype': df.dtypes,
        'missing': df.isnull().sum(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).round(2),
        'unique': df.nunique(),
        'sample': df.iloc[0],
    })
    return report.sort_values('missing_pct', ascending=False)
```

### Step 2: Missing Data Strategy
| Missing % | Strategy |
|-----------|---------|
| < 5% | Listwise deletion or simple imputation (mean/median) |
| 5-20% | Multiple imputation (MICE) or KNN imputation |
| 20-50% | Create missing indicator + impute, or domain-specific rules |
| > 50% | Consider dropping feature, or use as-is if informative |

### Step 3: Outlier Detection
Methods in order of preference:
1. Domain knowledge (physical/logical limits)
2. IQR method (1.5 * IQR beyond Q1/Q3)
3. Z-score (|z| > 3 for normal data)
4. Isolation Forest (multivariate outliers)
5. DBSCAN (density-based detection)

### Step 4: Data Validation
```python
import pandera as pa

schema = pa.DataFrameSchema({
    "age": pa.Column(int, pa.Check.in_range(0, 150)),
    "email": pa.Column(str, pa.Check.str_matches(r'^[\w.-]+@[\w.-]+\.\w+$')),
    "revenue": pa.Column(float, pa.Check.ge(0)),
    "category": pa.Column(str, pa.Check.isin(["A", "B", "C"])),
})

validated_df = schema.validate(df)
```

## Performance Optimization

### Pandas Performance Tips
1. Use `pd.read_csv(dtype=...)` to specify types upfront
2. Use `category` dtype for low-cardinality strings
3. Use `pd.eval()` and `df.query()` for complex expressions
4. Use `df.itertuples()` over `df.iterrows()` (10-100x faster)
5. Use vectorized operations over loops
6. Consider `polars` or `dask` for datasets > 1GB

### Memory Optimization
```python
def optimize_memory(df):
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype('category')
    return df
```

## Reporting Templates

### Analysis Report Structure
1. Executive Summary (1 paragraph)
2. Data Description (source, size, timeframe, quality)
3. Methodology (techniques used, assumptions, limitations)
4. Key Findings (3-5 bullet points with supporting visuals)
5. Recommendations (actionable next steps)
6. Appendix (detailed tables, supplementary charts)

### Metric Definitions Checklist
Always define metrics precisely before analysis:
- Numerator and denominator
- Time window
- Inclusion/exclusion criteria
- Aggregation method (mean, median, sum)
- Comparison baseline
