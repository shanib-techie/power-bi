import pandas as pd
df = pd.read_json("C:\\Users\\Admin\\OneDrive\\Desktop\\power bi\\deep_insight\\employees.json")
df.to_excel("C:\\Users\\Admin\\OneDrive\\Desktop\\power bi\\deep_insight\\employees.xlsx",index=False)
print(df)

