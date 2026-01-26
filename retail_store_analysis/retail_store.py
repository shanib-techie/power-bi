import pandas as pd

df = pd.read_excel("C:\\Users\\Admin\\OneDrive\\Desktop\\power bi\\retail_store_analysis\\Retail-Store-Transactions (1).xlsx")

print(df.head(4))
df.to_html("C:\\Users\\Admin\\OneDrive\\Desktop\\power bi\\retail_store_analysis\\Retail-Store-Transactions (1).html")