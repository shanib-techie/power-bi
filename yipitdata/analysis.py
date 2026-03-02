import pandas as pd

"""Q1: As of the most recent week appearing in the data set, which English title had the highest cumulative weeks in the top 10?
 What were the average weekly hours viewed of this title across the entire data set?
 Reminder: The week of May 22nd, 2022 experienced an outage that affected the 'weekly_hours_viewed' column. 
Due to this, we have input the missing values as 0 in the data file. For any calculations, you need to ignore the outage week."""

df = pd.read_excel("yipitdata\\Data File.xlsx")

print(df.describe())

df['week'] = pd.to_datetime(df['week'])
most_recent_week = df['week'].max()
recent_df = df[df['week'] == most_recent_week]
english_recent = recent_df[recent_df['category'].str.contains('English')]
top_title_row = english_recent.loc[english_recent['cumulative_weeks_in_top_10'].idxmax()]
top_title = top_title_row['show_title']
top_cum_weeks = top_title_row['cumulative_weeks_in_top_10']
outage_date = pd.to_datetime("2022-05-22")

title_df = df[(df['show_title'] == top_title) &(df['week'] != outage_date)]
average_hours = title_df['weekly_hours_viewed'].mean()
print("Most recent week:", most_recent_week)
print("Title:", top_title)
print("Cumulative weeks:", top_cum_weeks)
print("Average weekly hours viewed:", round(average_hours, 2))


"""2: What is the weekly rank of the lowest IMDb-rated title for the most recent week appearing in the data set? 
Please explain how you arrived at your answer. *"""

path = "C:\\Users\\Admin\\OneDrive\\Desktop\\power bi\\yipitdata\\Data File.xlsx"
df_weekly = pd.read_excel(path, sheet_name=0)   # first sheet
df_imdb = pd.read_excel(path, sheet_name=1)     # second sheet
print(df_weekly.columns)
print(df_imdb.columns)
# 1️⃣ Most recent week nikaalo
latest_week = df_weekly["week"].max()

# 2️⃣ Sirf latest week ka data filter karo
df_latest = df_weekly[df_weekly["week"] == latest_week]

# 3️⃣ Merge with IMDb ratings
df_merged = pd.merge(
    df_latest,
    df_imdb,
    left_on="show_title",
    right_on="title",
    how="left"
)
# 4️⃣ Lowest IMDb rating nikaalo
lowest_rating = df_merged["rating"].min()

# 5️⃣ Us title ka weekly rank nikaalo
result = df_merged[df_merged["rating"] == lowest_rating]

print(result[["show_title", "rating", "weekly_rank"]])