import pandas as pd
import plotly.express as px

file_path = input("Enter Excel file path (example: data/TC5_rates.xlsx): ").strip()
x_col = input("Enter x-axis column name: ").strip()
y_col = input("Enter y-axis column name: ").strip()

df = pd.read_excel(file_path)

# Convert to datetime
df[x_col] = pd.to_datetime(df[x_col], errors="coerce")

# Drop missing values
df = df.dropna(subset=[x_col, y_col])

# Plot
fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")

fig.update_traces(
    mode="lines",
    hovertemplate="Date: %{x|%d-%m-%Y}<br>Value: %{y}<extra></extra>"
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title=y_col,
    hovermode="x unified"
)

# Format x-axis display
fig.update_xaxes(
    tickformat="%d-%m-%Y"
)

fig.show()