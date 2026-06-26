import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# ===== USER INPUTS =====
independent_file = input("Enter the weekly independent variable file path: ").strip()
dependent_file = input("Enter the daily dependent variable file path: ").strip()

independent_date_col = "Date"
independent_value_col = "Value"

dependent_date_col = "Date"
dependent_value_col = "Value"

independent_label = input("Enter the independent variable label [Loadings]: ").strip() or "Loadings"
dependent_label = input("Enter the dependent variable label [WS_Rate]: ").strip() or "WS_Rate"

weekly_frequency = input("Enter weekly frequency [W-FRI]: ").strip() or "W-FRI"
weekly_method = input("Aggregate dependent variable by 'mean' or 'last' [mean]: ").strip().lower() or "mean"

lag_periods = int(input("Enter lag periods in weeks [0]: ").strip() or 0)
outlier_z_threshold = float(input("Enter residual z-score threshold [2.5]: ").strip() or 2.5)
remove_only_upper_outliers = input("Remove only upper outliers? (yes/no) [yes]: ").strip().lower() != "no"
# =======================

# Load files
independent_df = pd.read_csv(independent_file)
dependent_df = pd.read_csv(dependent_file)

# Keep only needed columns
independent_df = independent_df[[independent_date_col, independent_value_col]].copy()
dependent_df = dependent_df[[dependent_date_col, dependent_value_col]].copy()

# Rename columns
independent_df.columns = ["Date", independent_label]
dependent_df.columns = ["Date", dependent_label]

# Convert dates
independent_df["Date"] = pd.to_datetime(independent_df["Date"], errors="coerce")
dependent_df["Date"] = pd.to_datetime(dependent_df["Date"], errors="coerce")

# Drop missing
independent_df = independent_df.dropna()
dependent_df = dependent_df.dropna()

# Filter years
independent_df = independent_df[
    (independent_df["Date"].dt.year >= 2018) &
    (independent_df["Date"].dt.year <= 2026)
]
dependent_df = dependent_df[
    (dependent_df["Date"].dt.year >= 2018) &
    (dependent_df["Date"].dt.year <= 2026)
]

print("\nOriginal Data Info:")
print("Independent rows:", len(independent_df))
print("Dependent rows:", len(dependent_df))
print("Independent date range:", independent_df["Date"].min(), "to", independent_df["Date"].max())
print("Dependent date range:", dependent_df["Date"].min(), "to", dependent_df["Date"].max())

# ---------------------------
# RESAMPLE DEPENDENT VARIABLE TO WEEKLY
# ---------------------------
dependent_df = dependent_df.sort_values("Date")
dependent_df = dependent_df.set_index("Date")

if weekly_method == "last":
    dependent_weekly = dependent_df.resample(weekly_frequency).last().reset_index()
elif weekly_method == "mean":
    dependent_weekly = dependent_df.resample(weekly_frequency).mean().reset_index()
else:
    dependent_weekly = dependent_df.resample(weekly_frequency).first().reset_index()


# Make sure independent is sorted too
independent_df = independent_df.sort_values("Date")

print("\nWeekly Dependent Preview:")
print(dependent_weekly.head())

# ---------------------------
# MERGE WEEKLY SERIES
# ---------------------------
df = pd.merge(independent_df, dependent_weekly, on="Date", how="inner")
df = df.sort_values("Date")

print("\nMerged weekly rows:", len(df))

if df.empty:
    print("No overlapping weekly data found. Check weekly date alignment.")
    exit()

# Apply lag in weeks
shifted_col = f"{independent_label}_shifted"
df[shifted_col] = df[independent_label].shift(lag_periods)

# Drop invalid rows after lag
df = df.dropna(subset=[shifted_col, dependent_label])

if df.empty:
    print("No data left after applying lag.")
    exit()

print("\nMerged Weekly Data Preview:")
print(df.head())

# Correlation
corr = df[shifted_col].corr(df[dependent_label])
print(f"\nInitial weekly correlation between {independent_label} and {dependent_label}: {corr:.4f}")

# ---------------------------
# INITIAL REGRESSION
# ---------------------------
X_initial = sm.add_constant(df[[shifted_col]])
y_initial = df[dependent_label]

initial_model = sm.OLS(y_initial, X_initial).fit()

print("\nINITIAL WEEKLY REGRESSION SUMMARY (Before Outlier Removal):")
print(initial_model.summary())

# Add fitted values and residuals
df["Fitted_Initial"] = initial_model.predict(X_initial)
df["Residuals_Initial"] = y_initial - df["Fitted_Initial"]

# Residual z-score
resid_std = df["Residuals_Initial"].std()
df["Residual_Z"] = df["Residuals_Initial"] / resid_std

# ---------------------------
# FILTER OUTLIERS
# ---------------------------
if remove_only_upper_outliers:
    df_clean = df[df["Residual_Z"] <= outlier_z_threshold].copy()
else:
    df_clean = df[df["Residual_Z"].abs() <= outlier_z_threshold].copy()

print("\nOutlier Filtering:")
print("Original rows:", len(df))
print("Rows after filtering:", len(df_clean))
print("Rows removed:", len(df) - len(df_clean))

if df_clean.empty:
    print("No data left after outlier removal.")
    exit()

# ---------------------------
# CLEANED REGRESSION
# ---------------------------
X_clean = sm.add_constant(df_clean[[shifted_col]])
y_clean = df_clean[dependent_label]

clean_model = sm.OLS(y_clean, X_clean).fit()

print("\nCLEANED WEEKLY REGRESSION SUMMARY (After Outlier Removal):")
print(clean_model.summary())

# Add fitted values/residuals
df_clean["Fitted_Clean"] = clean_model.predict(X_clean)
df_clean["Residuals_Clean"] = y_clean - df_clean["Fitted_Clean"]

# Add year for coloring
df["Year"] = df["Date"].dt.year
df_clean["Year"] = df_clean["Date"].dt.year

# ---------------------------
# PLOT 1: ORIGINAL WEEKLY SCATTER + REGRESSION
# ---------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x=shifted_col, y=dependent_label, hue="Year", palette="tab10", alpha=0.8)
sns.regplot(data=df, x=shifted_col, y=dependent_label, scatter=False, line_kws={"color": "red"})
plt.title(f"Weekly {dependent_label} vs {independent_label} (Original Data)")
plt.xlabel(independent_label)
plt.ylabel(dependent_label)
plt.grid(True)
plt.tight_layout()

# ---------------------------
# PLOT 2: CLEANED WEEKLY SCATTER + REGRESSION
# ---------------------------
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x=shifted_col, y=dependent_label, hue="Year", palette="tab10", alpha=0.8)
sns.regplot(data=df_clean, x=shifted_col, y=dependent_label, scatter=False, line_kws={"color": "red"})
plt.title(f"Weekly {dependent_label} vs {independent_label} (Cleaned Data)")
plt.xlabel(independent_label)
plt.ylabel(dependent_label)
plt.grid(True)
plt.tight_layout()

# ---------------------------
# PLOT 3: ACTUAL VS FITTED (CLEANED)
# ---------------------------
plt.figure(figsize=(12, 6))
plt.plot(df_clean["Date"], df_clean[dependent_label], label="Actual", color="blue")
plt.plot(df_clean["Date"], df_clean["Fitted_Clean"], label="Fitted", color="red")
plt.title(f"Weekly Actual vs Fitted {dependent_label} (Cleaned)")
plt.xlabel("Date")
plt.ylabel(dependent_label)
plt.legend()
plt.grid(True)
plt.tight_layout()

# ---------------------------
# PLOT 4: BOTH SERIES OVER TIME (CLEANED)
# ---------------------------
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(df_clean["Date"], df_clean[dependent_label], color="blue", label=dependent_label)
ax1.set_xlabel("Date")
ax1.set_ylabel(dependent_label, color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

ax2 = ax1.twinx()
ax2.plot(df_clean["Date"], df_clean[shifted_col], color="green", label=independent_label)
ax2.set_ylabel(independent_label, color="green")
ax2.tick_params(axis="y", labelcolor="green")

plt.title(f"Weekly {dependent_label} and {independent_label} Over Time (Cleaned)")
fig.tight_layout()

plt.show()