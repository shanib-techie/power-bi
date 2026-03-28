import pandas as pd

# Load datasets
sales = pd.read_csv("crm_sales_dataset.csv")
reps = pd.read_csv("sales_reps.csv")
products = pd.read_csv("products.csv")

print(sales.head())
print(sales.info())




# 1. REMOVE DUPLICATES
sales = sales.drop_duplicates()

# 2. HANDLE MISSING VALUES

# Fill missing amount with median
sales['amount'] = sales['amount'].fillna(sales['amount'].median())


# 3. STANDARDIZE STAGE VALUES

sales['stage'] = sales['stage'].str.strip().str.title()


# 4. CONVERT DATE

sales['date'] = pd.to_datetime(sales['date'])


# 5. CREATE MAPPING (NAME → ID)

# Merge to get sales_rep_id
sales = sales.merge(reps, left_on='sales_rep', right_on='sales_rep_name', how='left')

# Merge to get product_id
sales = sales.merge(products, left_on='product', right_on='product_name', how='left')


# 6. FINAL CLEAN DATASET

final_data = sales[['deal_id', 'sales_rep_id', 'product_id', 'amount', 'stage', 'date']]

# Save cleaned data
final_data.to_csv("cleaned_sales_data.csv", index=False)

print(" Cleaning Done & File Saved ")