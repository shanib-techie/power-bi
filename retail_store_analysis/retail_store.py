import pandas as pd


df = pd.read_excel("C:\\Users\\Admin\\OneDrive\\Desktop\\power bi\\retail_store_analysis\\Retail-Store-Transactions (1).xlsx")


df.drop(columns = ["Unnamed: 17"], inplace = True)
# df.drop(columns= ["bonus"], inplace = True)
print(df.head(4))

# df.drop("Unnamed: 14")
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("how much time each product sell",df["Product"].value_counts())
print(df["Product"].unique())
print("how much time each location come",df["Location"].value_counts())
print(df["Location"].unique())
print("how much time store manager sales",df["StoreManager"].value_counts())
print(df["StoreManager"].unique())
print("cashier",df["Cashier"].unique())
print(df["Cashier"].value_counts())
print(max(df["TotalPrice"]))


print("WHERE LOCATION IS store C")
print("count of the following ",len(df[(df["Location"] == "Store C")  & (df["Product"] == "Tablet")]))
print(df[(df["Location"] == "Store C")  & (df["Product"] == "Tablet")])

print(df)

# print(df)
df.to_html("C:\\Users\\Admin\\OneDrive\\Desktop\\power bi\\retail_store_analysis\\Retail-Store-Transactions (1).html")