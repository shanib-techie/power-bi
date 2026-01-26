import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

input_file_path = "fruit_classification_dataset.csv"

output_file_path = "fruit_classification_dataset.csv"

# Load the dataset
df=pd.read_csv(input_file_path)
# Assuming data cleaning has been done as per Practical-1_CSC-24-56.py

data = df.copy()
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(data.select_dtypes(include=[np.number])), columns=data.select_dtypes(include=[np.number]).columns)

# Add a new column with log-transformed 'Price' (if 'Price' exists)
if 'Price' in data_scaled.columns:
    data_scaled['log_Price'] = np.log1p(data_scaled['Price'])
    data_scaled.drop(columns=['Price'], inplace=True)

# Display the first few rows of the scaled data
print(data_scaled.head())

# Aggregate the scaled data by 'Manufacturer' and calculate mean of each feature
if 'Manufacturer' in data.columns:
    aggregated_data = data_scaled.copy()
    aggregated_data['Manufacturer'] = data['Manufacturer'].values
    manufacturer_agg = aggregated_data.groupby('Manufacturer').mean()
    print(manufacturer_agg)
else:
    print("Column 'Manufacturer' not found in the dataset.")

# Discretize the 'log_Price' column into 4 bins (quartiles) if it exists
if 'log_Price' in data_scaled.columns:
    data_scaled['log_Price_bin'] = pd.qcut(data_scaled['log_Price'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    print(data_scaled[['log_Price', 'log_Price_bin']].head())
else:
    print("'log_Price' column not found for discretization.")

# Randomly sample 10 rows from the scaled data
sampled_data = data_scaled.sample(n=10, random_state=42)
print("Randomly sampled data:")
print(sampled_data)



# Save the processed data to a new CSV file
data_scaled.to_csv(output_file_path, index=False)