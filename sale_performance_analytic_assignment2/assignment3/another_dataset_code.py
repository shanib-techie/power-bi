import pandas as pd
import random
from datetime import datetime, timedelta

# Reproducibility
random.seed(42)

# Entities
sales_reps = ["Amit", "Riya", "Karan", "Neha"]
regions = ["North", "South", "East", "West"]
products = ["Product A", "Product B", "Product C"]

# Product pricing
product_prices = {
    "Product A": 5000,
    "Product B": 6000,
    "Product C": 7000
}

# Start date
start_date = datetime(2024, 1, 1)

data = []

for i in range(792):
    date = start_date + timedelta(days=i)

    month = date.month

    # 🔥 Seasonality (Q2 peak)
    if month in [1, 2]:
        leads = random.randint(40, 70)
    elif month in [3, 4, 5]:
        leads = random.randint(80, 120)
    elif month in [6, 7]:
        leads = random.randint(60, 90)
    else:
        leads = random.randint(50, 80)

    # Choose attributes
    rep = random.choice(sales_reps)
    region = random.choice(regions)
    product = random.choice(products)

    # Opportunities conversion
    opportunities = int(leads * random.uniform(0.5, 0.75))

    # 🔥 Rep performance differences
    if rep == "Amit":
        deals_closed = int(opportunities * random.uniform(0.5, 0.65))  # top
    elif rep == "Riya":
        deals_closed = int(opportunities * random.uniform(0.4, 0.55))
    elif rep == "Karan":
        deals_closed = int(opportunities * random.uniform(0.25, 0.4))  # weak
    else:  # Neha
        deals_closed = int(opportunities * random.uniform(0.35, 0.5))

    # 🔥 Region impact (North strong)
    if region == "North":
        deals_closed = int(deals_closed * random.uniform(1.1, 1.3))
    elif region == "West":
        deals_closed = int(deals_closed * random.uniform(0.9, 1.1))
    elif region == "South":
        deals_closed = int(deals_closed * random.uniform(0.85, 1.0))
    else:  # East
        deals_closed = int(deals_closed * random.uniform(0.8, 0.95))

    # Product pricing
    deal_size = product_prices[product]

    # Revenue
    revenue = deals_closed * deal_size

    # Units sold (variation)
    units_sold = deals_closed * random.randint(1, 3)

    # 🔥 Sales cycle (better reps close faster)
    if rep == "Amit":
        sales_cycle = random.randint(10, 15)
    elif rep == "Riya":
        sales_cycle = random.randint(12, 18)
    elif rep == "Karan":
        sales_cycle = random.randint(18, 25)
    else:
        sales_cycle = random.randint(14, 20)

    data.append([
        date.strftime("%Y-%m-%d"),
        rep,
        region,
        product,
        leads,
        opportunities,
        deals_closed,
        revenue,
        units_sold,
        deal_size,
        sales_cycle
    ])

# Columns
columns = [
    "Date", "Sales_Rep", "Region", "Product",
    "Leads", "Opportunities", "Deals_Closed",
    "Revenue", "Units_Sold", "Deal_Size", "Sales_Cycle_Days"
]

df = pd.DataFrame(data, columns=columns)

# Save CSV
df.to_csv("sales_dataset_792.csv", index=False)

print("🔥 Dataset Ready: sales_dataset_792.csv")