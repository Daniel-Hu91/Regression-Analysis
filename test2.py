import pandas as pd

oil_file = "data/TC5_rates.xlsx"   # change to your file

# Show available sheets
excel_file = pd.ExcelFile(oil_file)
print("Sheets:", excel_file.sheet_names)

# Read first sheet for inspection
df = pd.read_excel(oil_file)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 10 rows:")
print(df.head(10))

print("\nData types:")
print(df.dtypes)