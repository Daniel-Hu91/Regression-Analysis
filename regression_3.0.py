import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import mplcursors

# ===== USER INPUTS =====
independent_file = input("Enter the independent variable file path: ").strip()
dependent_file =  input("Enter the dependent variable file path: ").strip()

independent_date_col = "Date"
independent_value_col = "Value"

dependent_date_col = "Date"
dependent_value_col = "Value"

independent_label = input("Enter the independent variable label: ").strip()
dependent_label = input("Enter the dependent variable label: ").strip()

lag_periods = 0
outlier_z_threshold = 0.5   # try 2.0, 2.5, or 3.0
remove_only_upper_outliers = True
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

# Drop missing rows
independent_df = independent_df.dropna()
dependent_df = dependent_df.dropna()

# Filter years
independent_df = independent_df[
    (independent_df["Date"].dt.year >= 2018) &
    (independent_df["Date"].dt.year <= 2025)
]
dependent_df = dependent_df[
    (dependent_df["Date"].dt.year >= 2018) &
    (dependent_df["Date"].dt.year <= 2025)
]
print("Independent rows:", len(independent_df))
print("Dependent rows:", len(dependent_df))
print("Independent date range:", independent_df["Date"].min(), "to", independent_df["Date"].max())
print("Dependent date range:", dependent_df["Date"].min(), "to", dependent_df["Date"].max())

# Merge on date
df = pd.merge(independent_df, dependent_df, on="Date", how="inner")
df = df.sort_values("Date")

print("Merged rows after exact date merge:", len(df))

# Apply lag
shifted_col = f"{independent_label}_shifted"
df[shifted_col] = df[independent_label].shift(lag_periods)

# Drop rows made invalid by shifting
df = df.dropna(subset=[shifted_col, dependent_label])

print("\nMerged Data Preview:")
print(df.head())

print("\nInitial Correlation:")
print(df[shifted_col].corr(df[dependent_label]))

# ---------------------------
# INITIAL REGRESSION
# ---------------------------
X_initial = sm.add_constant(df[[shifted_col]])
y_initial = df[dependent_label]

initial_model = sm.OLS(y_initial, X_initial).fit()

print("\nINITIAL REGRESSION SUMMARY (Before Outlier Removal):")
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

# ---------------------------
# CLEANED REGRESSION
# ---------------------------
X_clean = sm.add_constant(df_clean[[shifted_col]])
y_clean = df_clean[dependent_label]

clean_model = sm.OLS(y_clean, X_clean).fit()

print("\nCLEANED REGRESSION SUMMARY (After Outlier Removal):")
print(clean_model.summary())

# Add fitted values/residuals for cleaned data
df_clean["Fitted_Clean"] = clean_model.predict(X_clean)
df_clean["Residuals_Clean"] = y_clean - df_clean["Fitted_Clean"]

# ---------------------------
# PLOT 1: ORIGINAL SCATTER + REGRESSION
# ---------------------------
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x=shifted_col, y=dependent_label, line_kws={"color": "red"})
plt.title(f"{dependent_label} vs {independent_label} (Original Data)")
plt.xlabel(independent_label)
plt.ylabel(dependent_label)
plt.grid(True)
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

# ---------------------------
# PLOT 2: CLEANED SCATTER + REGRESSION
# ---------------------------
plt.figure(figsize=(10, 6))
sns.regplot(data=df_clean, x=shifted_col, y=dependent_label, line_kws={"color": "red"})
plt.title(f"{dependent_label} vs {independent_label} (Residual Outliers Removed)")
plt.xlabel(independent_label)
plt.ylabel(dependent_label)
plt.grid(True)
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

# ---------------------------
# PLOT 3: RESIDUALS BEFORE FILTERING
# ---------------------------
plt.figure(figsize=(10, 6))
plt.scatter(df[shifted_col], df["Residuals_Initial"], alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.axhline(outlier_z_threshold * resid_std, color="orange", linestyle="--", label="Upper Threshold")
if not remove_only_upper_outliers:
    plt.axhline(-outlier_z_threshold * resid_std, color="orange", linestyle="--", label="Lower Threshold")
plt.title("Initial Residuals vs Independent Variable")
plt.xlabel(independent_label)
plt.ylabel("Residual")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

# ---------------------------
# PLOT 4: ACTUAL VS FITTED (ORIGINAL)
# ---------------------------
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df[dependent_label], label="Actual", color="blue")
plt.plot(df["Date"], df["Fitted_Initial"], label="Fitted", color="red")
plt.title(f"Actual vs Fitted {dependent_label} (Original Data)")
plt.xlabel("Date")
plt.ylabel(dependent_label)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

# ---------------------------
# PLOT 5: ACTUAL VS FITTED (CLEANED)
# ---------------------------
plt.figure(figsize=(12, 6))
plt.plot(df_clean["Date"], df_clean[dependent_label], label="Actual", color="blue")
plt.plot(df_clean["Date"], df_clean["Fitted_Clean"], label="Fitted", color="red")
plt.title(f"Actual vs Fitted {dependent_label} (Cleaned Data)")
plt.xlabel("Date")
plt.ylabel(dependent_label)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

# ---------------------------
# PLOT 6: RESIDUAL PLOT AFTER FILTERING
# ---------------------------
plt.figure(figsize=(8, 6))
plt.scatter(df_clean["Fitted_Clean"], df_clean["Residuals_Clean"], alpha=0.7)
plt.axhline(0, color="red", linestyle="--")
plt.title("Residual Plot After Filtering")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.grid(True)
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)


fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(df_clean["Date"], df_clean[dependent_label], color="blue", label=dependent_label)
ax1.set_xlabel("Date")
ax1.set_ylabel(dependent_label, color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

ax2 = ax1.twinx()
ax2.plot(df_clean["Date"], df_clean[shifted_col], color="red", label=independent_label)
ax2.set_ylabel(independent_label, color="red")
ax2.tick_params(axis="y", labelcolor="red")

plt.title(f"{dependent_label} and {independent_label} Over Time (Cleaned Data)")
fig.tight_layout()
plt.show(block=False)
plt.pause(0.1)

input("Press Enter to close all plots...")