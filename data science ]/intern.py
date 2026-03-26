import pandas as pd
import numpy as np

df = pd.read_csv("C:\\Users\\Admin\\OneDrive\\Desktop\\power bi\\data science ]\\fear_greed_index.csv")

print(df)
print(df.describe())
# count_of_classification = df["classification"]

print("count of each claassification")
print(df["classification"].value_counts())

where_val_more_39_60_less = df[(df["value"] > 39) & (df["value"] < 60)]
print("counting ",where_val_more_39_60_less["classification"].value_counts())
print(len(where_val_more_39_60_less))

print(where_val_more_39_60_less)