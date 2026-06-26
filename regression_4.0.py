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

# Drop missing rows
independent_df = independent_df.dropna()
dependent_df = dependent_df.dropna()

# Filter years
independent_df = independent_df[
    (independent_df["Date"].dt.year >= 2018)
]
dependent_df = dependent_df[
    (dependent_df["Date"].dt.year >= 2018)
]

# Merge on date
df_all = pd.merge(independent_df, dependent_df, on="Date", how="inner")
df_all = df_all.sort_values("Date")

print("Merged rows after exact date merge:", len(df_all))

if df_all.empty:
    print("No overlapping data found.")
    raise SystemExit

years = sorted(df_all["Date"].dt.year.unique())

# Turn on interactive mode so figures can all stay open
plt.ion()

for year in years:
    print("\n" + "=" * 100)
    print(f"YEAR: {year}")
    print("=" * 100)

    df = df_all[df_all["Date"].dt.year == year].copy()

    if len(df) < 10:
        print(f"Skipping {year}: not enough observations.")
        continue

    shifted_col = f"{independent_label}_shifted"
    df[shifted_col] = df[independent_label].shift(lag_periods)
    df = df.dropna(subset=[shifted_col, dependent_label])

    if len(df) < 10:
        print(f"Skipping {year}: not enough observations after lag.")
        continue

    # ---------------------------
    # ORIGINAL REGRESSION
    # ---------------------------
    X_initial = sm.add_constant(df[[shifted_col]])
    y_initial = df[dependent_label]
    initial_model = sm.OLS(y_initial, X_initial).fit()

    print("\nORIGINAL REGRESSION SUMMARY:")
    print(initial_model.summary())

    # Residuals for outlier filtering
    df["Fitted_Initial"] = initial_model.predict(X_initial)
    df["Residuals_Initial"] = y_initial - df["Fitted_Initial"]

    resid_std = df["Residuals_Initial"].std()

    if resid_std == 0 or pd.isna(resid_std):
        print(f"Skipping {year}: residual std is zero or invalid.")
        continue

    df["Residual_Z"] = df["Residuals_Initial"] / resid_std

    # ---------------------------
    # OUTLIER FILTERING
    # ---------------------------
    if remove_only_upper_outliers:
        df_clean = df[df["Residual_Z"] <= outlier_z_threshold].copy()
    else:
        df_clean = df[df["Residual_Z"].abs() <= outlier_z_threshold].copy()

    print("\nOUTLIER FILTERING:")
    print("Original rows:", len(df))
    print("Rows after filtering:", len(df_clean))
    print("Rows removed:", len(df) - len(df_clean))
    print("Percent removed:", round((len(df) - len(df_clean)) / len(df) * 100, 2), "%")

    if len(df_clean) < 10:
        print(f"Skipping cleaned regression for {year}: not enough observations after filtering.")
        continue

    # ---------------------------
    # CLEANED REGRESSION
    # ---------------------------
    X_clean = sm.add_constant(df_clean[[shifted_col]])
    y_clean = df_clean[dependent_label]
    clean_model = sm.OLS(y_clean, X_clean).fit()

    print("\nCLEANED REGRESSION SUMMARY:")
    print(clean_model.summary())

    # ---------------------------
    # ORIGINAL SCATTER PLOT
    # ---------------------------
    plt.figure(figsize=(10, 6))
    sns.regplot(
        data=df,
        x=shifted_col,
        y=dependent_label,
        scatter_kws={"alpha": 0.65, "s": 30},
        line_kws={"color": "red"}
    )
    plt.title(
        f"{year} Original\n"
        f"R²={initial_model.rsquared:.3f}, Adj R²={initial_model.rsquared_adj:.3f}, "
        f"Coef={initial_model.params[shifted_col]:.4f}, p={initial_model.pvalues[shifted_col]:.4g}"
    )
    plt.xlabel(independent_label)
    plt.ylabel(dependent_label)
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    # ---------------------------
    # CLEANED SCATTER PLOT
    # ---------------------------
    plt.figure(figsize=(10, 6))
    sns.regplot(
        data=df_clean,
        x=shifted_col,
        y=dependent_label,
        scatter_kws={"alpha": 0.65, "s": 30},
        line_kws={"color": "red"}
    )
    plt.title(
        f"{year} Cleaned\n"
        f"R²={clean_model.rsquared:.3f}, Adj R²={clean_model.rsquared_adj:.3f}, "
        f"Coef={clean_model.params[shifted_col]:.4f}, p={clean_model.pvalues[shifted_col]:.4g}"
    )
    plt.xlabel(independent_label)
    plt.ylabel(dependent_label)
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

input("Press Enter to close all plots...")