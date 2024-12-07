import pandas as pd

# Load the data
data = pd.read_csv("data.csv")

# Print column names to check if 'runs' and 'wickets' exist
print(data.columns)

# Strip spaces in column names if any
data.columns = data.columns.str.strip()

# Now check if 'runs' and 'wickets' are in the columns
if 'runs' in data.columns and 'wickets' in data.columns:
    print("Columns found!")
else:
    print("Columns not found!")
