import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# ===== USER INPUTS =====
independent_file = input("Enter the independent variable file path: ").strip()
dependent_file = input("Enter the dependent variable file path: ").strip()

independent_date_col = "Date"
independent_value_col = "Value"

dependent_date_col = "Date"
dependent_value_col = "Value"

independent_label = input("Enter the independent variable label [OoW]: ").strip() or "OoW"
dependent_label = input("Enter the dependent variable label [WS_Rate]: ").strip() or "WS_Rate"

lag_periods = int(input("Enter lag periods [0]: ").strip() or 0)
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
    (independent_df["Date"].dt.year <= 2026)
]
dependent_df = dependent_df[
    (dependent_df["Date"].dt.year >= 2018) &
    (dependent_df["Date"].dt.year <= 2026)
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

# Create daily differences
delta_x_col = f"Delta_{independent_label}"
delta_y_col = f"Delta_{dependent_label}"

df[delta_x_col] = df[shifted_col].diff()
df[delta_y_col] = df[dependent_label].diff()

# Drop rows made invalid by differencing
df = df.dropna(subset=[delta_x_col, delta_y_col])

print("\nDifferenced Data Preview:")
print(df[["Date", shifted_col, dependent_label, delta_x_col, delta_y_col]].head())

print("\nCorrelation of daily changes:")
print(df[delta_x_col].corr(df[delta_y_col]))

# ---------------------------
# DIFFERENCE REGRESSION
# ---------------------------
X = sm.add_constant(df[[delta_x_col]])
y = df[delta_y_col]

model = sm.OLS(y, X).fit()

print("\nDIFFERENCE REGRESSION SUMMARY:")
print(model.summary())

# Add fitted values and residuals
df["Fitted_Delta"] = model.predict(X)
df["Residuals_Delta"] = y - df["Fitted_Delta"]

# ---------------------------
# PLOT 1: Scatter of daily changes
# ---------------------------
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x=delta_x_col, y=delta_y_col, line_kws={"color": "red"})
plt.title(f"Daily Change in {dependent_label} vs Daily Change in {independent_label}")
plt.xlabel(f"Daily Change in {independent_label}")
plt.ylabel(f"Daily Change in {dependent_label}")
plt.grid(True)
plt.tight_layout()

# ---------------------------
# PLOT 2: Daily changes over time
# ---------------------------
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(df["Date"], df[delta_y_col], color="blue", label=delta_y_col)
ax1.set_xlabel("Date")
ax1.set_ylabel(delta_y_col, color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

ax2 = ax1.twinx()
ax2.plot(df["Date"], df[delta_x_col], color="green", label=delta_x_col)
ax2.set_ylabel(delta_x_col, color="green")
ax2.tick_params(axis="y", labelcolor="green")

plt.title(f"Daily Changes: {dependent_label} and {independent_label}")
fig.tight_layout()

# ---------------------------
# PLOT 3: Actual vs fitted change
# ---------------------------
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df[delta_y_col], label="Actual Daily Change", color="blue")
plt.plot(df["Date"], df["Fitted_Delta"], label="Fitted Daily Change", color="red")
plt.title(f"Actual vs Fitted Daily Change in {dependent_label}")
plt.xlabel("Date")
plt.ylabel(f"Daily Change in {dependent_label}")
plt.legend()
plt.grid(True)
plt.tight_layout()

# ---------------------------
# PLOT 4: Residual plot
# ---------------------------
plt.figure(figsize=(8, 6))
plt.scatter(df["Fitted_Delta"], df["Residuals_Delta"], alpha=0.7)
plt.axhline(0, color="red", linestyle="--")
plt.title("Residual Plot for Difference Regression")
plt.xlabel("Fitted Daily Change")
plt.ylabel("Residual")
plt.grid(True)
plt.tight_layout()

# Show all plots at once
plt.show()