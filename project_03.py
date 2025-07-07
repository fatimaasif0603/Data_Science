import pandas as pd

# Load CSV with proper settings
df = pd.read_csv('car_data.csv', encoding='latin1', on_bad_lines='skip')

# Clean column names (remove spaces)
df.columns = df.columns.str.strip()

# Convert numeric columns to correct types (optional but helpful)
numeric_cols = ['Year', 'Selling_Price', 'Present_Price', 'Driven_kms', 'Owner']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Preview the data
print(df.head())
print(df.info())
