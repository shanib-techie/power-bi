import pandas as pd
df = pd.read_json("C:\\Users\\Admin\\OneDrive\\Desktop\\power bi\\deep_insight\\employees.json")
df.to_html("C:\\Users\\Admin\\OneDrive\\Desktop\\power bi\\deep_insight\\employees.html",index=False)
print(df)

