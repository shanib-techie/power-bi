# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# df = pd.read_csv("C:\\Users\\Admin\\OneDrive\\Desktop\\power bi\\sale_performance_analytic_assignment2\\assignment3\\sales_dataset_500.csv")
# print(df.head())
# print(df.info())
# print(df["Product"].value_counts())
# print(df["Region"].value_counts())
# print(df["Sales_Rep"].value_counts())

# grph = df["Sales_Rep"].value_counts()
# grph.plot(kind="bar")
# plt.ylabel("count")
# plt.xlabel("SALES_REP")
# plt.grid("-")
# # plt.show()

# grp_by_sale_with_region = df.groupby(["Sales_Rep","Region"])["Revenue"].sum()
# print(grp_by_sale_with_region)


# grp_by_sale_with_region_and_product_also = df.groupby(["Sales_Rep","Region","Product"])["Revenue"].sum()
# print(grp_by_sale_with_region_and_product_also)


# result_of_each_sale_rep = df.groupby("Sales_Rep").agg(
#     total_revenue=("Revenue","sum"),
#     average_revenue=("Revenue","mean"),
#     min_revenue=("Revenue","min"),
#     count_revenue=("Revenue","count")


# )
# print(result_of_each_sale_rep)

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("C:\\Users\\Admin\\OneDrive\\Desktop\\power bi\\sale_performance_analytic_assignment2\\assignment3\\sales_dataset_792.csv")

# Convert Date column
df['Date'] = pd.to_datetime(df['Date'])

# Sort data (important)
df = df.sort_values('Date')

# Monthly Revenue Aggregation
monthly = df.groupby(pd.Grouper(key='Date', freq='ME'))['Revenue'].sum().reset_index()

# Drop any NaN months
monthly = monthly.dropna()

# Create time index
monthly['Time_Index'] = np.arange(len(monthly))

# Define features and target
X = monthly[['Time_Index']]
y = monthly['Revenue']

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# FUTURE FORECAST (Next 6 months AFTER March 2026)
future_index = np.arange(len(monthly), len(monthly) + 6).reshape(-1,1)
forecast = model.predict(future_index)

# Correct future dates
last_date = monthly['Date'].max()

future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=6,
    freq='ME'
)

# Create forecast dataframe
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecast_Revenue': forecast
})

# PLOT
plt.figure(figsize=(10,5))

# Actual data
plt.plot(monthly['Date'], monthly['Revenue'], label='Actual Revenue')

# Forecast data
plt.plot(forecast_df['Date'], forecast_df['Forecast_Revenue'], linestyle='dashed', label='Forecast')

# Labels
plt.title("Sales Revenue Trend & Forecast (Next 2 Quarters)")
plt.xlabel("Date")
plt.ylabel("Revenue")

plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()