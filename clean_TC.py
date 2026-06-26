import pandas as pd

# Ask user for file paths
file_path = input("Enter the Excel file path to clean: ").strip()
output_path = input("Enter the output CSV file path: ").strip()

# Read Excel file
df = pd.read_excel(file_path)

# Rename the 2nd and 7th columns to Date and Value
df = df.rename(columns={
    df.columns[1]: "Date",
    df.columns[6]: "Value"
})

# Use renamed column names directly
date_col = "Date"
value_col = "Value"

# Convert Date column to datetime
df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# Drop rows with missing Date or Value
df = df.dropna(subset=[date_col, value_col])

# Keep 2018 through 2025 only
df = df[(df[date_col] >= "2018-01-01")]

# Sort oldest first
df = df.sort_values(date_col)

# Save cleaned file
df.to_csv(output_path, index=False)

print("\nCleaned preview:")
print(df.head())
print(df.tail())
print("Shape:", df.shape)
print(f"Cleaned file saved to: {output_path}")