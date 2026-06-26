import pandas as pd
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go

# ===== USER INPUTS =====
independent_file = "data/Loadings_Cleaned.csv"
dependent_file = "data/TC5_Cleaned.csv"

independent_date_col = "Date"
independent_value_col = "Value"

dependent_date_col = "Date"
dependent_value_col = "Value"

lag_periods = 0
exclude_years = [2023]
loadings_scale = 100000
hac_maxlags = 5
# =======================

# Load files
independent_df = pd.read_csv(independent_file)
dependent_df = pd.read_csv(dependent_file)

# Clean column names
independent_df.columns = independent_df.columns.astype(str).str.strip()
dependent_df.columns = dependent_df.columns.astype(str).str.strip()

print("Independent file columns:", independent_df.columns.tolist())
print("Dependent file columns:", dependent_df.columns.tolist())

# Keep only needed columns
independent_df = independent_df[[independent_date_col, independent_value_col]].copy()
dependent_df = dependent_df[[dependent_date_col, dependent_value_col]].copy()

# Rename for consistency
independent_df.columns = ["Date", "Loadings"]
dependent_df.columns = ["Date", "WS Rate"]

# Convert types
independent_df["Date"] = pd.to_datetime(independent_df["Date"], errors="coerce")
dependent_df["Date"] = pd.to_datetime(dependent_df["Date"], errors="coerce")
independent_df["Loadings"] = pd.to_numeric(independent_df["Loadings"], errors="coerce")
dependent_df["WS Rate"] = pd.to_numeric(dependent_df["WS Rate"], errors="coerce")

# Drop missing rows
independent_df = independent_df.dropna()
dependent_df = dependent_df.dropna()

# Exclude selected years
if exclude_years:
    independent_df = independent_df[~independent_df["Date"].dt.year.isin(exclude_years)]
    dependent_df = dependent_df[~dependent_df["Date"].dt.year.isin(exclude_years)]

# Merge on date
df = pd.merge(independent_df, dependent_df, on="Date", how="inner")

# Sort by date
df = df.sort_values("Date").reset_index(drop=True)

# Apply lag to loadings
df["Loadings_shifted"] = df["Loadings"].shift(lag_periods)

# Scale loadings for interpretability
df["Loadings_scaled"] = df["Loadings_shifted"] / loadings_scale

# Lagged dependent variable
df["WS_Rate_lag1"] = df["WS Rate"].shift(1)

# First differences
df["d_WS_Rate"] = df["WS Rate"].diff()
df["d_Loadings_scaled"] = df["Loadings_scaled"].diff()

# Drop rows invalid for main model
df_main = df.dropna(subset=["Loadings_scaled", "WS Rate"]).copy()

print("\nMerged Data Preview:")
print(df_main.head())

print("\nMerged Data Info:")
print(df_main.info())

print("\nDate range:")
print("Start:", df_main["Date"].min())
print("End:", df_main["Date"].max())
print("Rows:", len(df_main))

# =========================
# MODEL 1: SIMPLE OLS
# WS Rate ~ Loadings
# =========================
X1 = sm.add_constant(df_main["Loadings_scaled"])
y1 = df_main["WS Rate"]

model1 = sm.OLS(y1, X1).fit(cov_type="HAC", cov_kwds={"maxlags": hac_maxlags})

print("\nMODEL 1: SIMPLE OLS")
print(model1.summary())

# =========================
# MODEL 2: WITH LAGGED WS RATE
# WS Rate ~ Loadings + WS Rate lag1
# =========================
df_model2 = df.dropna(subset=["Loadings_scaled", "WS Rate", "WS_Rate_lag1"]).copy()

X2 = df_model2[["Loadings_scaled", "WS_Rate_lag1"]]
X2 = sm.add_constant(X2)
y2 = df_model2["WS Rate"]

model2 = sm.OLS(y2, X2).fit(cov_type="HAC", cov_kwds={"maxlags": hac_maxlags})

print("\nMODEL 2: OLS WITH LAGGED WS RATE")
print(model2.summary())

# =========================
# MODEL 3: DIFFERENCED MODEL
# d(WS Rate) ~ d(Loadings)
# =========================
df_model3 = df.dropna(subset=["d_WS_Rate", "d_Loadings_scaled"]).copy()

X3 = sm.add_constant(df_model3["d_Loadings_scaled"])
y3 = df_model3["d_WS_Rate"]

model3 = sm.OLS(y3, X3).fit(cov_type="HAC", cov_kwds={"maxlags": hac_maxlags})

print("\nMODEL 3: DIFFERENCED OLS")
print(model3.summary())

# =========================
# MODEL COMPARISON TABLE
# =========================
comparison_df = pd.DataFrame([
    {
        "Model": "1. WS Rate ~ Loadings",
        "Rows": len(df_main),
        "R2": model1.rsquared,
        "Adj_R2": model1.rsquared_adj,
        "AIC": model1.aic,
        "BIC": model1.bic
    },
    {
        "Model": "2. WS Rate ~ Loadings + WS Rate lag1",
        "Rows": len(df_model2),
        "R2": model2.rsquared,
        "Adj_R2": model2.rsquared_adj,
        "AIC": model2.aic,
        "BIC": model2.bic
    },
    {
        "Model": "3. d(WS Rate) ~ d(Loadings)",
        "Rows": len(df_model3),
        "R2": model3.rsquared,
        "Adj_R2": model3.rsquared_adj,
        "AIC": model3.aic,
        "BIC": model3.bic
    }
])

print("\nMODEL COMPARISON:")
print(comparison_df)

comparison_df.to_csv("data/regression_model_comparison.csv", index=False)
print("\nSaved model comparison to data/regression_model_comparison.csv")

# =========================
# LAG COMPARISON
# =========================
print("\nLAG COMPARISON:")
lag_results = []

for lag in [0, 1, 2, 3, 5, 10]:
    temp = pd.merge(independent_df, dependent_df, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    temp["Loadings_shifted"] = temp["Loadings"].shift(lag)
    temp["Loadings_scaled"] = temp["Loadings_shifted"] / loadings_scale
    temp = temp.dropna(subset=["Loadings_scaled", "WS Rate"])

    X_lag = sm.add_constant(temp["Loadings_scaled"])
    y_lag = temp["WS Rate"]

    lag_model = sm.OLS(y_lag, X_lag).fit(cov_type="HAC", cov_kwds={"maxlags": hac_maxlags})

    lag_results.append({
        "Lag": lag,
        "Rows": len(temp),
        "Coefficient": lag_model.params["Loadings_scaled"],
        "PValue": lag_model.pvalues["Loadings_scaled"],
        "R2": lag_model.rsquared,
        "Adj_R2": lag_model.rsquared_adj,
        "AIC": lag_model.aic,
        "BIC": lag_model.bic
    })

lag_results_df = pd.DataFrame(lag_results)
print(lag_results_df)

lag_results_df.to_csv("data/lag_comparison_results.csv", index=False)
print("\nSaved lag comparison to data/lag_comparison_results.csv")

# =========================
# INTERPRETATION PRINTS
# =========================
print("\nINTERPRETATION:")
print(f"Model 1 coefficient: {model1.params['Loadings_scaled']:.4f}")
print(f"This means a {loadings_scale:,} increase in loadings is associated with a {model1.params['Loadings_scaled']:.4f} change in WS Rate.")

if "Loadings_scaled" in model2.params.index:
    print(f"Model 2 coefficient on Loadings: {model2.params['Loadings_scaled']:.4f}")

if "d_Loadings_scaled" in model3.params.index:
    print(f"Model 3 coefficient on change in Loadings: {model3.params['d_Loadings_scaled']:.4f}")

# =========================
# PLOT 1: SCATTER + TRENDLINE
# =========================
fig1 = px.scatter(
    df_main,
    x="Loadings_scaled",
    y="WS Rate",
    title=f"WS Rate vs Loadings (Lag={lag_periods}, scale={loadings_scale:,})",
    trendline="ols",
    hover_data={
        "Date": True,
        "Loadings": ":,.2f",
        "Loadings_shifted": ":,.2f",
        "Loadings_scaled": ":,.2f",
        "WS Rate": ":,.2f"
    }
)

fig1.update_traces(
    hovertemplate=(
        "Date: %{customdata[0]|%d-%m-%Y}<br>"
        f"Loadings / {loadings_scale:,}: %{x:.2f}<br>"
        "WS Rate: %{y:.2f}<extra></extra>"
    )
)

fig1.update_layout(
    xaxis_title=f"Loadings / {loadings_scale:,}",
    yaxis_title="WS Rate"
)

fig1.show()

# =========================
# PLOT 2: TIME SERIES
# =========================
fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=df_main["Date"],
    y=df_main["WS Rate"],
    mode="lines",
    name="WS Rate",
    hovertemplate="Date: %{x|%d-%m-%Y}<br>WS Rate: %{y:.2f}<extra></extra>"
))

fig2.add_trace(go.Scatter(
    x=df_main["Date"],
    y=df_main["Loadings_scaled"],
    mode="lines",
    name=f"Loadings / {loadings_scale:,}",
    yaxis="y2",
    hovertemplate=f"Date: %{{x|%d-%m-%Y}}<br>Loadings / {loadings_scale:,}: %{{y:.2f}}<extra></extra>"
))

fig2.update_layout(
    title="WS Rate and Loadings Over Time",
    xaxis=dict(title="Date"),
    yaxis=dict(title="WS Rate", side="left"),
    yaxis2=dict(
        title=f"Loadings / {loadings_scale:,}",
        overlaying="y",
        side="right"
    ),
    hovermode="x unified"
)

fig2.show()

# =========================
# PLOT 3: DIFFERENCED SCATTER
# =========================
fig3 = px.scatter(
    df_model3,
    x="d_Loadings_scaled",
    y="d_WS_Rate",
    title="Change in WS Rate vs Change in Loadings",
    trendline="ols",
    hover_data={
        "Date": True,
        "d_Loadings_scaled": ":,.4f",
        "d_WS_Rate": ":,.4f"
    }
)

fig3.update_traces(
    hovertemplate=(
        "Date: %{customdata[0]|%d-%m-%Y}<br>"
        f"Change in Loadings / {loadings_scale:,}: %{x:.4f}<br>"
        "Change in WS Rate: %{y:.4f}<extra></extra>"
    )
)

fig3.update_layout(
    xaxis_title=f"Change in Loadings / {loadings_scale:,}",
    yaxis_title="Change in WS Rate"
)

fig3.show()