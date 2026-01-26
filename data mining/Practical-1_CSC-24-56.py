import pandas as pd
import numpy as np


# Load the dataset
df=pd.read_csv("fruit_classification_dataset.csv")
# Initial exploration
print("First 5 rows:")
print(df.head())

# Data types and summary statistics
# print("\nData types of columns:")
# print(df.dtypes)

# # Summary statistics and missing values
# print("\nSummary statistics:")
# print(df.describe(include='all'))
# print("\nMissing values in each column:")
# print(df.isnull().sum())

# # If missing values are few, drop rows; otherwise, fill with appropriate values
# threshold = 0.05 * len(df)  # 5% of total rows

# # Handling missing values
# for col in df.columns:
#     missing_count = df[col].isnull().sum()
#     if missing_count > 0:
#         if missing_count < threshold:
#             df = df.dropna(subset=[col])
#             print(f"Dropped rows with missing values in '{col}'.")
#         else:
#             if df[col].dtype == 'object':
#                 df[col] = df[col].fillna(df[col].mode()[0])
#                 print(f"Filled missing values in '{col}' with mode.")
#             else:
#                 df[col] = df[col].fillna(df[col].median())
#                 print(f"Filled missing values in '{col}' with median.")

# # Check for duplicates
# print("\nUnique values in categorical columns:")
# categorical_cols = df.select_dtypes(include=['object']).columns
# for col in categorical_cols:
#     print(f"\nColumn '{col}':")
#     print(df[col].unique())

# # Data validation rules
# # Rule 1: 'Year of manufacture' should be between 1980 and current year
# current_year = pd.Timestamp.now().year
# if 'Year of manufacture' in df.columns:
#     invalid_years = ~df['Year of manufacture'].between(1980, current_year)
#     print(f"\nRows with invalid 'Year of manufacture': {df[invalid_years].shape[0]}")
#     print(df[invalid_years][['Year of manufacture']])

# # Rule 2: 'Mileage' and 'Price' should be positive numbers
# for col in ['Mileage', 'Price']:
#     if col in df.columns:
#         invalid = df[col] <= 0
#         print(f"\nRows with non-positive '{col}': {df[invalid].shape[0]}")
#         print(df[invalid][[col]])

# # Rule 3: 'Engine size' should be between 0.6 and 8.0 liters (example range)
# if 'Engine size' in df.columns:
#     invalid_engine = ~df['Engine size'].between(0.6, 8.0)
#     print(f"\nRows with unrealistic 'Engine size': {df[invalid_engine].shape[0]}")
#     print(df[invalid_engine][['Engine size']])

# # After identifying issues, you can choose to drop or correct them
# df.to_csv(output_file_path, index=False)
# print(f"\nCleaned data saved to '{output_file_path}'.")