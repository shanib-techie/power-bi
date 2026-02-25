import pandas as pd
import matplotlib.pyplot as plt

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
# print(max(df["TotalPrice"]))






print("WHERE LOCATION IS store C and product is tablet")
print("count of the following ",len(df[(df["Location"] == "Store C")  & (df["Product"] == "Tablet")]))
print(df[(df["Location"] == "Store C")  & (df["Product"] == "Tablet")])


r_from_c = df[df["Location"] == "Store C"]["TotalPrice"].sum()
print("total revenue from c store",r_from_c)


r_from_d = df[df["Location"] == "Store D"]["TotalPrice"].sum()
print("revenue from d location",r_from_d)


print("revenue from each location ")
r_each_l = df.groupby("Location")["TotalPrice"].sum()
print(r_each_l)







tablet_revenue = df[df["Product"] == "Tablet"]["TotalPrice"].sum()
print("How much revenue from Tablet:", tablet_revenue)
printer_revenue = df[df["Product"] == "Printer"]["TotalPrice"].sum()
print("how much revenue from printer",printer_revenue)
phone_revenue = df[df["Product"] == "Phone"]["TotalPrice"].sum()
print("how much revenue from phone",phone_revenue)
monitor_revenue = df[df["Product"] == "Monitor"]["TotalPrice"].sum()
print("revenue from monitor ",monitor_revenue)
laptop_revenue = df[df["Product"] == "Laptop"]["TotalPrice"].sum()
print("revenue from Laptop",laptop_revenue)
chair_revenue = df[df["Product"] == "Chair"]["TotalPrice"].sum()
print("revenue from chair ",chair_revenue)
desk_revenue = df[df["Product"] == "Desk"]["TotalPrice"].sum()
print("REvenue from desk",desk_revenue)



r_each_p = df.groupby("Product")["TotalPrice"].sum()
print("revenue of each product",r_each_p)






print(df)
revenue = df.groupby(["Location", "Product"])["TotalPrice"].sum()#kaam ka function h

print("revenue from ech location with every product",revenue)


# ANYLSIS ON  PaymentType
print("PAYMENT METHOD ==>",df["PaymentType"].unique())
print("count of payment method ")
print(df["PaymentType"].value_counts())
# kis loacation se kitne kitne payment type hue h 
loc_with_payment_type = df.groupby(["Location","PaymentType"])["TotalPrice"].sum()
print("each location with count of payment method ",loc_with_payment_type)
loc_with_payment_type_count = df.groupby(["Location","PaymentType"]).size()
print(loc_with_payment_type_count)

# loc_with_payment_type_count_make_graph = df.groupby(["Location","PaymentType"]).size().unstack()

# loc_with_payment_type_count_make_graph.plot(kind="bar")

# # plt.xlabel("Stores")
# plt.ylabel("Count of Payment Type")
# plt.title("Payment Type Count per Store")

# plt.legend(title="Payment Type")
# plt.show()
# print(df)
df.to_html("C:\\Users\\Admin\\OneDrive\\Desktop\\power bi\\retail_store_analysis\\Retail-Store-Transactions (1).html")

# operation on store manager

storemanager_names = df["StoreManager"].unique()
print("NUMBER OF STORE MANAGER = > ",len(storemanager_names))
print("store manager = >",storemanager_names)

# each store manager total sale 
print("each store manager total sale :")
val_of_each_store_manager = df["StoreManager"].value_counts()
print(val_of_each_store_manager)

print("DID CASHIER AND STORRE MANAGER ARE SAME ??")
grp_of_cashier_and_manager = df.groupby(["Cashier","StoreManager"]).size()
print(grp_of_cashier_and_manager)

# grp_of_cashier_and_manager_graph = df.groupby(["Cashier","StoreManager"]).size().unstack()

# grp_of_cashier_and_manager_graph.plot(kind="bar")
# plt.title("cashier group with manager")
# plt.grid("-")
# plt.ylabel("count of cashier with differnet store manager ")
# plt.show()





grp_of_manager_with_product = df.groupby(["StoreManager" , "Product"])["TotalPrice"].sum()
print("STORE MANAGER WITH HIS SALES IN EACH PRODUCT : ")
print(grp_of_manager_with_product)



print(df["Cashier"].unique())
# ///////////////////////////////GROUP GRAPH OF EACH STORE MANAGER PERTICULAR ITEM SALES
# grp_of_manager_with_product_grap = df.groupby(["StoreManager" , "Product"])["TotalPrice"].sum().unstack()

# grp_of_manager_with_product_grap.plot(kind = "bar")
# plt.grid()
# plt.xlabel("store manager ")
# plt.show()





# ////////////////////////////////////////bar graph of each storemanager sales count
# plt.bar(val_of_each_store_manager.index,val_of_each_store_manager.values,color="yellow",label="store manager sales")
# plt.xlabel("store manager")
# plt.ylabel("store manager count of sales")
# plt.show()



# pie chart of store manager with its sale
sale_of_each_storemaneger = df.groupby(["StoreManager"])["TotalPrice"].sum()
print(sale_of_each_storemaneger)
plt.pie(sale_of_each_storemaneger,labels=sale_of_each_storemaneger.index,autopct="%1.1f%%",colors=["red","yellow","pink","green","blue"])#label  pie ki per seace ki valy
plt.grid()
plt.title("SALES OF EACH STORE MANAGER ")
plt.show()
