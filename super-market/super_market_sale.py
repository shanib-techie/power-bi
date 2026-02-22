import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("super-market\\super-market-sale_data_after_operation.xlsx")
# Step 1: clean column names
df.columns = df.columns.astype(str)

# Step 2: remove junk excel columns
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

df.to_html("super-market\\super-market-sale_data_after_operation.html")
print(df.describe())


print(df)