import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\Admin\\OneDrive\\Desktop\\power bi\\sale_performance_analytic_assignment2\\assignment3\\sales_dataset_500.csv")
print(df.head())
print(df.info())
print(df["Product"].value_counts())
print(df["Region"].value_counts())
print(df["Sales_Rep"].value_counts())

grph = df["Sales_Rep"].value_counts()
grph.plot(kind="bar")
plt.ylabel("count")
plt.xlabel("SALES_REP")
plt.grid("-")
# plt.show()

grp_by_sale_with_region = df.groupby(["Sales_Rep","Region"])["Revenue"].sum()
print(grp_by_sale_with_region)


grp_by_sale_with_region_and_product_also = df.groupby(["Sales_Rep","Region","Product"])["Revenue"].sum()
print(grp_by_sale_with_region_and_product_also)


result_of_each_sale_rep = df.groupby("Sales_Rep").agg(
    total_revenue=("Revenue","sum"),
    average_revenue=("Revenue","mean"),
    min_revenue=("Revenue","min"),
    count_revenue=("Revenue","count")


)
print(result_of_each_sale_rep)