import sys

import pandas as pd

df = pd.read_csv(sys.argv[1])

# Check for missing data per column
missing_data = df.isnull().sum()
missing_percentage = df.isnull().mean() * 100

# Combine both into a single DataFrame for better visualisation
missing_info = pd.DataFrame({'Missing Values': missing_data, 'Percentage Missing (%)': missing_percentage})

# Output the results to a CSV file
missing_info.to_csv('missing_data_info.csv')
print("Missing data information has been saved to 'missing_data_info.csv'.")

# Print the total number of rows
print(f"Number of rows: {df.shape[0]}")

# Drop rows with missing values
complete_cases_df = df.dropna()
print(f"Number of rows after complete case analysis (removing all rows with any missing value): {complete_cases_df.shape[0]}")

percentage_lost = ((df.shape[0] - complete_cases_df.shape[0]) / df.shape[0]) * 100
print(f"Percentage of rows lost due to complete case analysis: {percentage_lost:.2f}%")