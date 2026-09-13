"""
Titanic Dataset — Data Preprocessing, EDA & Visualization
Intern Project: Data Cleaning + Storytelling with Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 110

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv("data/titanic_raw.csv")
print("Raw shape:", df.shape)

# ---------------------------------------------------------------
# 2. HANDLE DUPLICATES
# ---------------------------------------------------------------
n_dupes = df.duplicated().sum()
print(f"Duplicate rows found: {n_dupes}")
df = df.drop_duplicates().reset_index(drop=True)
print("Shape after removing duplicates:", df.shape)

# ---------------------------------------------------------------
# 3. HANDLE MISSING VALUES
# ---------------------------------------------------------------
print("\nMissing values before cleaning:\n", df.isnull().sum()[df.isnull().sum() > 0])

# age -> fill with median (robust to skew)
df["age"] = df["age"].fillna(df["age"].median())

# embarked / embark_town -> fill with mode (most common port)
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])

# deck has ~77% missing -> too sparse to impute meaningfully, drop column
df = df.drop(columns=["deck"])

print("\nMissing values after cleaning:\n", df.isnull().sum().sum(), "total missing values remain")

# ---------------------------------------------------------------
# 4. HANDLE OUTLIERS (fare) using IQR method
# ---------------------------------------------------------------
Q1 = df["fare"].quantile(0.25)
Q3 = df["fare"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df["fare"] < lower) | (df["fare"] > upper)]
print(f"\nFare outliers detected: {len(outliers)} (bounds: {lower:.2f} to {upper:.2f})")

# Cap outliers instead of dropping, to preserve sample size
df["fare_capped"] = df["fare"].clip(lower=lower, upper=upper)

# ---------------------------------------------------------------
# 5. QUICK SUMMARY STATS
# ---------------------------------------------------------------
summary = df.describe(include="all").T
summary.to_csv("outputs/summary_statistics.csv")
df.to_csv("outputs/titanic_cleaned.csv", index=False)
print("\nCleaned dataset and summary statistics saved to outputs/")

# ---------------------------------------------------------------
# 6. VISUALIZATIONS (individual charts)
# ---------------------------------------------------------------

# 6.1 Missing values heatmap (before cleaning, using raw df for illustration)
raw = pd.read_csv("data/titanic_raw.csv")
plt.figure(figsize=(8, 5))
sns.heatmap(raw.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Map (Raw Data)")
plt.tight_layout()
plt.savefig("outputs/01_missing_values_heatmap.png")
plt.close()

# 6.2 Boxplot: fare before vs after outlier capping
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(x=df["fare"], ax=axes[0], color="salmon")
axes[0].set_title("Fare — Before Capping")
sns.boxplot(x=df["fare_capped"], ax=axes[1], color="lightgreen")
axes[1].set_title("Fare — After Capping")
plt.tight_layout()
plt.savefig("outputs/02_fare_outliers.png")
plt.close()

# 6.3 Survival count by sex
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="sex", hue="alive")
plt.title("Survival Count by Sex")
plt.tight_layout()
plt.savefig("outputs/03_survival_by_sex.png")
plt.close()

# 6.4 Survival rate by passenger class
plt.figure(figsize=(6, 4))
sns.barplot(data=df, x="pclass", y="survived", hue="pclass", palette="Set2", legend=False)
plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig("outputs/04_survival_by_class.png")
plt.close()

# 6.5 Age distribution
plt.figure(figsize=(7, 4))
sns.histplot(df["age"], bins=30, kde=True, color="steelblue")
plt.title("Age Distribution")
plt.tight_layout()
plt.savefig("outputs/05_age_distribution.png")
plt.close()

# 6.6 Correlation heatmap (numeric columns)
plt.figure(figsize=(7, 6))
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/06_correlation_heatmap.png")
plt.close()

# ---------------------------------------------------------------
# 7. DASHBOARD (combined multi-panel summary)
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Titanic Dataset — Key Findings Dashboard", fontsize=16, fontweight="bold")

sns.countplot(data=df, x="sex", hue="alive", ax=axes[0, 0])
axes[0, 0].set_title("Survival by Sex")

sns.barplot(data=df, x="pclass", y="survived", hue="pclass", palette="Set2", legend=False, ax=axes[0, 1])
axes[0, 1].set_title("Survival Rate by Class")

sns.histplot(df["age"], bins=25, kde=True, ax=axes[0, 2], color="steelblue")
axes[0, 2].set_title("Age Distribution")

sns.boxplot(x=df["fare_capped"], ax=axes[1, 0], color="lightgreen")
axes[1, 0].set_title("Fare Distribution (Cleaned)")

sns.countplot(data=df, x="embark_town", hue="survived", ax=axes[1, 1])
axes[1, 1].set_title("Survival by Embarkation Town")
axes[1, 1].tick_params(axis='x', rotation=15)

sns.heatmap(numeric_df.corr(), annot=True, fmt=".1f", cmap="coolwarm", ax=axes[1, 2], cbar=False)
axes[1, 2].set_title("Correlation Heatmap")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("outputs/07_dashboard.png", dpi=130)
plt.close()

print("\nAll charts saved to outputs/. Dashboard: outputs/07_dashboard.png")
