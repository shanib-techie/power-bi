
print("total revenue from c store",r_from_c)


r_from_d = df[df["Location"] == "Store D"]["TotalPrice"].sum()
print("revenue from d location",r_from_d)