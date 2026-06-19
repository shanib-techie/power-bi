import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_excel("C:\\Users\\Admin\\OneDrive\\Desktop\\power bi\\STUDENT_data\\student_data.xlsx")
# print(df.describe())
# print(df.columns)
# stutus = df["status"].value_counts()
# print(stutus)
# schools_names = df["SchoolName"].unique()
# print(schools_names)

# school_count = df["SchoolName"].value_counts()
# print(school_count)
# group_of_school_name_school_id = df.groupby(["SchoolName", "School ID"])[["SchoolName","School ID"]].count()
# print(group_of_school_name_school_id)


# ===========SCHOOL state district and school_name count=========================
schl_state_district_name = df.groupby(["State","Distict","SchoolName"])["State"].value_counts()
print(schl_state_district_name)
# ===================DELHI SCHOOL COUNT BY DISTRICT ====================================
schl_in_delhi = df[(df["State"] == "Delhi")][["State","Distict","SchoolName"]].value_counts()
# print(schl_in_delhi)

# schl_in_delhi = df[(df["State"] == "Delhi")][["Distict","SchoolName"]].value_counts().plot(kind="bar")
# plt.xticks(rotation=79)
# plt.show()
# 
# # =========iss state ke kitne distict =========================
# no_of_distict_by_state_val = df.groupby("State")["Distict"].unique()

# print(no_of_distict_by_state_val)

# no_of_distict_by_state = df.groupby(["State","Distict"])[["State","Distict"]].count()
# print(no_of_distict_by_state)
# print(df["State"].value_counts())
# print(df["Distict"].value_counts())


# # =======================State Distict SchoolName ==============================

# grp_sta_dis_schlN = pd.crosstab(
# [df["State"] , df["Distict"]] ,
#  df["SchoolName"],
#  margins=True
# )

# print(grp_sta_dis_schlN)

# print(df["Start Date"].equals(df["End Date"]))


# print(df["testCode"].is_unique)

# for col in df.columns:
#     print(col, ":", df[col].is_unique)

max_score_row = df[(df["percentScore"] == df["percentScore"].max())]["State"] 
print(max_score_row)

print("TOP 50 PERCENT")
# top50 = df["percentScore"].sort_values(ascending=False).head(50)  yeh wala tab jab direct print ke udner likho
top_50 = df.sort_values(by="percentScore", ascending=False).head(50)
print(top_50[["State","SchoolName","percentScore"]])
top_50["State"].value_counts().plot(kind="bar")
# plt.xticks(rotation=270)
# plt.show()
print(top_50["State"].value_counts())



state_distict_opper = df.pivot_table(
    values="percentScore",
    index="Distict",
    columns="State",
    aggfunc=min
)

print(state_distict_opper)