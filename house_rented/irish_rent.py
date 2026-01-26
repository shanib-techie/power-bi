import pandas as pd
df = pd.read_csv("C:\\Users\\Admin\\OneDrive\\Desktop\\power bi\\house_rented\\irish_rent_by_county.csv")
df.to_excel("C:\\Users\\Admin\\OneDrive\\Desktop\\power bi\\house_rented\\irish_rent_by_county.xlsx")
# print(df)
# "C:\Users\Admin\OneDrive\Desktop\power bi\deep_insight\employees.xlsx"
print("some statistic thing :  \n")
print (df.describe())# numeric ka samaan return krta h



print("shape : return M X N : ")
print(df.shape)

print("total no of coloum : ")
print(df.columns)


# return true where null
print(df.isnull())
# BEDROOM_NUM is true so its null
"""NOW, WE STUDY DIFFERENT WAY TO DEAL WITH THIS TYPE OF PROBLEM"""
# df.dropna(inplace = True)
df.fillna(0,inplace= True)#agr starting me 0 krde so missing ki jagah 0 hoga
print(df)
# print(df.isnull())

# #  AGR HAME FILE ME CHANE KRNE H TOH NYI FILE BANADO LAST ME USME AUTOMATICALLY CHNAGE DIK  JAINHE
# import pandas as pd

# df = pd.read_excel("data.xlsx")

# # operations
# df["Total"] = df["Sales"] + df["Profit"]

# # NEW file me save
# df.to_excel("data_updated.xlsx", index=False)
