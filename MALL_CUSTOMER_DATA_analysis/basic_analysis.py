import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\Admin\\OneDrive\\Desktop\\power bi\\data mining\\Mall_Customers.csv")

print(df.head(10))

print(df.describe())



print("NUMBER OF MALE AND FEMALE")
print("MAle = ",(df["Gender"] == "Male").sum())
print("FEMALE = ",(df["Gender"] == "Female").sum())


group_where_male_score_less_35 = df[(df["Gender"] == "Male") & (df["Spending Score (1-100)"] <= 35)][["Gender", "Spending Score (1-100)"]]
print("NUMBER OF MALE CUSTOMER WHO LIES LESS THAN 35 SCORE AND MALE : ",len(group_where_male_score_less_35))
print(group_where_male_score_less_35)

group_where_male_score_more_70 = df[(df["Gender"] == "Male") & (df["Spending Score (1-100)"] >= 70)][["Gender" , "Spending Score (1-100)"]]
print("NUMBER OF MALE CUSTOMER WHO SCORE MORE THAN 70 ",len(group_where_male_score_more_70))
print(group_where_male_score_more_70)


group_where_female_score_less_35 = df[(df["Gender"] == "Female") & (df["Spending Score (1-100)"] <= 35)][["Gender", "Spending Score (1-100)"]]
print("NUMBER OF FEMALE CUSTOMER WHO LIES LESS THAN 35 SCORE AND MALE : ",len(group_where_female_score_less_35))
print(group_where_female_score_less_35)

group_where_female_score_more_70 = df[(df["Gender"] == "Female") & (df["Spending Score (1-100)"] >= 70)][["Gender", "Spending Score (1-100)"]]
print("NUMBER OF FEMALE CUSTOMER WHO SCORE MORE THAN 70 : ",len(group_where_female_score_more_70))
print(group_where_female_score_more_70)


print("NUMBER OF THE CUSTOMER WHO DIDNOT LIE LESS THAN 35 AND NOT MORE THAN 70")

group_of_that_customer_who_lies_between_35_and_70  = df[(df["Spending Score (1-100)"] > 35) & (df["Spending Score (1-100)"] < 70)]
print("NUMBER OF THE CUSTOMER WHO DIDNOT LIE LESS THAN 35 AND NOT MORE THAN 70",len(group_of_that_customer_who_lies_between_35_and_70))
print(group_of_that_customer_who_lies_between_35_and_70)


print("FINAL CROSS CHECK OF ALL CUSTOMER ",len(group_of_that_customer_who_lies_between_35_and_70) + len(group_where_female_score_less_35)+len(group_where_male_score_less_35)+len(group_where_female_score_more_70)+len(group_where_male_score_more_70))


print("RICH AND VALUED CUSTOMERS")
grp_age_bt_23_55_score_75_above_income_55_above = df[ (df["Age"] >= 23) & (df["Age"] <= 55 )  & (df["Spending Score (1-100)"] >= 75) & (df["Annual Income (k$)"] > 55) ]
print("NUMBER OF OUR GOLDEN CUSTOMER : ",len(grp_age_bt_23_55_score_75_above_income_55_above))
female_from_rich_cat = grp_age_bt_23_55_score_75_above_income_55_above[grp_age_bt_23_55_score_75_above_income_55_above["Gender"] == "Female"]
print("NUMBER OF FEMALE FROM THIS CLUSTER : " ,len(female_from_rich_cat))
male_from_rich_cat = grp_age_bt_23_55_score_75_above_income_55_above[grp_age_bt_23_55_score_75_above_income_55_above["Gender"] == "Male"]
print("NUMBER OF MALE FROM THIS CLUSTER  :",len(male_from_rich_cat))


print(grp_age_bt_23_55_score_75_above_income_55_above["Gender"].value_counts())

print(grp_age_bt_23_55_score_75_above_income_55_above)
