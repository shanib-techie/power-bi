import pandas as pd

df = pd.read_excel("C:\\Users\\Admin\\OneDrive\\Desktop\\power bi\\STUDENT_data\\student_data.xlsx")
print(df.describe())
print(df.columns)
stutus = df["status"].value_counts()
print(stutus)
schools_names = df["SchoolName"].unique()
print(schools_names)

school_count = df["SchoolName"].value_counts()
print(school_count)
group_of_school_name_school_id = df.groupby(["SchoolName", "School ID"])[["SchoolName","School ID"]].count()
print(group_of_school_name_school_id)


# ===========SCHOOL state district and school_name count=========================
schl_state_district_name = df.groupby(["State","Distict","SchoolName"])["State"].value_counts()
print(schl_state_district_name)

# =========iss state ke kitne distict =========================
no_of_distict_by_state_val = df.groupby("State")["Distict"].unique()

print(no_of_distict_by_state_val)

no_of_distict_by_state = df.groupby(["State","Distict"])[["State","Distict"]].count()
print(no_of_distict_by_state)
print(df["State"].value_counts())
print(df["Distict"].value_counts())