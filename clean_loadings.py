import pandas as pd

file_path = "data/ME_Loadings_Weekly.xlsx"
output_path = "data/ME_Loadings_Weekly_Cleaned.csv"

df = pd.read_excel(file_path)

df = df.rename(columns={
    df.columns[0]: "Date",
    df.columns[1]: "Value"
})


date_col = df.columns[0]
value_col = df.columns[1]

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

df = df.dropna(subset=[date_col, value_col])

# Keep 2018 through 2025 only
df = df[(df[date_col] >= "2018-01-01") & (df[date_col] < "2026-01-01")]

# Sort oldest first
df = df.sort_values(date_col)

df.to_csv(output_path, index=False)
df.to_csv("data/Loadings_Cleaned.csv", index=False)

print("Loadings cleaned preview:")
print(df.head())
print(df.tail())
print("Shape:", df.shape)