import pandas as pd

# ===== SETTINGS =====
INPUT_FILE = "data\SEAsia_SupplyvsDemand.xlsx"
OUTPUT_FILE = "data\SEAsia_SupplyvsDemand.csv"
DATE_COLUMN = "Date"
VALUE_COLUMN = "Value"
# ====================

def excel_to_csv(input_file, output_file, date_column, value_column, sheet_name=0):
    # Read Excel file
    df = pd.read_excel(input_file, sheet_name=sheet_name)

    # Check that columns exist
    missing = [col for col in [date_column, value_column] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Excel file: {missing}")

    # Keep only the two requested columns
    df = df[[date_column, value_column]]

    # Remove rows where both columns are blank
    df = df.dropna(how='all')

    # Convert date column to datetime
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    # Remove rows where date is blank/invalid
    df = df.dropna(subset=[date_column])

    # Sort earliest to latest
    df = df.sort_values(by=date_column)

    # Export to CSV
    df.to_csv(output_file, index=False)

    print(f"Done. CSV saved to: {output_file}")

excel_to_csv(INPUT_FILE, OUTPUT_FILE, DATE_COLUMN, VALUE_COLUMN)