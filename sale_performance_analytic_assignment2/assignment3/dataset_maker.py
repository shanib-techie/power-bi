import pandas as pd
import random
from datetime import datetime, timedelta

# Options
sales_reps = ["Amit", "Riya", "Karan", "Neha"]
regions = ["North", "South", "East", "West"]
products = ["Product A", "Product B", "Product C"]

# Generate dates
start_date = datetime(2024, 1, 1)

data = []

for i in range(500):
    date = start_date + timedelta(days=random.randint(0, 180))
    
    leads = random.randint(40, 100)
    opportunities = int(leads * random.uniform(0.5, 0.7))
    deals_closed = int(opportunities * random.uniform(0.3, 0.5))
    
    deal_size = random.choice([5000, 6000])
    revenue = deals_closed * deal_size
    units_sold = deals_closed * random.randint(2, 3)
    sales_cycle = random.randint(10, 25)
    
    row = [
        date.strftime("%Y-%m-%d"),
        random.choice(sales_reps),
        random.choice(regions),
        random.choice(products),
        leads,
        opportunities,
        deals_closed,
        revenue,
        units_sold,
        deal_size,
        sales_cycle
    ]
    
    data.append(row)

columns = ["Date","Sales_Rep","Region","Product","Leads","Opportunities","Deals_Closed","Revenue","Units_Sold","Deal_Size","Sales_Cycle_Days"]

df = pd.DataFrame(data, columns=columns)

df.to_csv("sales_dataset_500.csv", index=False)

print("Dataset created successfully!")