import pandas as pd
import numpy as np

df = pd.read_excel("C:\\Users\\Admin\\OneDrive\\Desktop\\power bi\\forage_dataset_task_to_visualization\\Online Retail.xlsx")


print("description of dataset:")
print(df.describe())


print("COLUMNS UNIQUES")

print(df["Country"].unique())

print(df["Country"].value_counts())

df["revenue"] = df["UnitPrice"] * df["Quantity"]

df.to_excel("updated_Online Retail.xlsx")

print(df.head())
