import pandas as pd
df = pd.read_csv("New folder\\Data.Gov_-_FY25_Q3.csv")
print(df)
df.to_html("New folder\\Data.Gov_-_FY25_Q3.html")