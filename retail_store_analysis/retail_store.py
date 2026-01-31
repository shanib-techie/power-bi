import pandas as pd

df = pd.read_excel("C:\\Users\\Admin\\OneDrive\\Desktop\\power bi\\retail_store_analysis\\Retail-Store-Transactions (1).xlsx")

df.to_html("C:\\Users\\Admin\\OneDrive\\Desktop\\power bi\\retail_store_analysis\\Retail-Store-Transactions (1).html")

df.drop(columns = ["Unnamed: 17"], inplace = True)
# df.drop(columns= ["bonus"], inplace = True)
print(df.head(4))
print(df["Product"].unique())
print(df["Location"].unique())
