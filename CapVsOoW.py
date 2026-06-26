import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# ===== USER INPUT =====
file_path = input("Enter the file path: ").strip()

date_col = "Date"
x_col = "Cap Diff"
y_col = "Loadings BPD"
# ======================

# Load file
df = pd.read_excel(file_path)

# Clean column names
df.columns = df.columns.str.strip()

print("\nAvailable columns:")
print(df.columns.tolist())

# Check required columns exist
required_cols = [date_col, x_col, y_col]
for col in required_cols:
    if col not in df.columns:
        print(f"Missing required column: {col}")
        raise SystemExit

# Keep needed columns only
df = df[[date_col, x_col, y_col]].copy()

# Convert date
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# Convert numeric columns
df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
df[y_col] = pd.to_numeric(df[y_col], errors="coerce")

# Drop missing rows
df = df.dropna(subset=[date_col, x_col, y_col])

# Sort by date
df = df.sort_values(date_col)

print("\nData preview:")
print(df.head())

print("\nData info:")
print(df.info())

print("\nDate range:")
print(df[date_col].min(), "to", df[date_col].max())

print("\nCorrelation:")
print(df[x_col].corr(df[y_col]))

# ---------------------------
# REGRESSION
# ---------------------------
X = sm.add_constant(df[[x_col]])
y = df[y_col]

model = sm.OLS(y, X).fit()

print("\nREGRESSION SUMMARY:")
print(model.summary())

# Add fitted values and residuals
df["Fitted"] = model.predict(X)
df["Residuals"] = y - df["Fitted"]

# ---------------------------
# PLOT 1: SCATTER + REGRESSION LINE
# ---------------------------
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x=x_col, y=y_col, line_kws={"color": "red"})
plt.title(f"{y_col} vs {x_col}")
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.grid(True)
plt.tight_layout()

# ---------------------------
# PLOT 2: ACTUAL VS FITTED OVER TIME
# ---------------------------
plt.figure(figsize=(12, 6))
plt.plot(df[date_col], df[y_col], label="Actual OoW BBL", color="blue")
plt.plot(df[date_col], df["Fitted"], label="Fitted OoW BBL", color="red")
plt.title(f"Actual vs Fitted {y_col} Over Time")
plt.xlabel("Date")
plt.ylabel(y_col)
plt.legend()
plt.grid(True)
plt.tight_layout()

# ---------------------------
# PLOT 3: RESIDUAL PLOT
# ---------------------------
plt.figure(figsize=(8, 6))
plt.scatter(df["Fitted"], df["Residuals"], alpha=0.7)
plt.axhline(0, color="red", linestyle="--")
plt.title("Residual Plot")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.grid(True)
plt.tight_layout()

# ---------------------------
# PLOT 4: BOTH SERIES OVER TIME
# ---------------------------
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(df[date_col], df[y_col], color="blue", label=y_col)
ax1.set_xlabel("Date")
ax1.set_ylabel(y_col, color="blue")
ax1.tick_params(axis="y", labelcolor="blue")

ax2 = ax1.twinx()
ax2.plot(df[date_col], df[x_col], color="green", label=x_col)
ax2.set_ylabel(x_col, color="green")
ax2.tick_params(axis="y", labelcolor="green")

plt.title(f"{y_col} and {x_col} Over Time")
fig.tight_layout()

plt.show()