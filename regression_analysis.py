import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# ===== USER INPUTS =====
independent_file = input("Enter the independent variable file path: ").strip()
dependent_file =  input("Enter the dependent variable file path: ").strip()

independent_date_col = "Date"
independent_value_col = "Value"

dependent_date_col = "Date"
dependent_value_col = "Value"

lag_periods = 10
# =======================

# Load files
independent_df = pd.read_csv(independent_file)
dependent_df = pd.read_csv(dependent_file)

# Keep only needed columns
independent_df = independent_df[[independent_date_col, independent_value_col]].copy()
dependent_df = dependent_df[[dependent_date_col, dependent_value_col]].copy()

# Rename for consistency
independent_df.columns = ["Date", "Loadings"]
dependent_df.columns = ["Date", "WS Rate"]

# Convert dates
independent_df["Date"] = pd.to_datetime(independent_df["Date"], errors="coerce")
dependent_df["Date"] = pd.to_datetime(dependent_df["Date"], errors="coerce")

# Drop missing rows
independent_df = independent_df.dropna()
dependent_df = dependent_df.dropna()

# Remove year 2023
independent_df = independent_df[independent_df["Date"].dt.year != 2023]
dependent_df = dependent_df[dependent_df["Date"].dt.year != 2023]

# Merge on date
df = pd.merge(independent_df, dependent_df, on="Date", how="inner")

# Sort by date
df = df.sort_values("Date")

# Apply lag
df["Loadings_shifted"] = df["Loadings"].shift(lag_periods)

# Drop rows made invalid by shifting
df = df.dropna(subset=["Loadings_shifted", "WS Rate"])

print("\nMerged Data Preview:")
print(df.head())

print("\nMerged Data Info:")
print(df.info())

df["Loadings_100k"] = df["Loadings_shifted"] / 100000

X = df["Loadings_100k"]
y = df["WS Rate"]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# Scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x="Loadings_shifted", y="WS Rate", line_kws={"color": "red"})
plt.title("Freight Rate vs Loadings")
plt.xlabel("Loadings")
plt.ylabel("Freight Rate")
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: time series plot
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(df["Date"], df["WS Rate"], color="blue", label="Freight Rate")
ax1.set_xlabel("Date")
ax1.set_ylabel("WS Rate", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

ax2 = ax1.twinx()
ax2.plot(df["Date"], df["Loadings_shifted"], color="green", label="Loadings")
ax2.set_ylabel("Loadings", color="green")
ax2.tick_params(axis="y", labelcolor="green")

plt.title("Freight Rate and Loadings Over Time")
fig.tight_layout()
plt.show()