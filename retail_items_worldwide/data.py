import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# df = pd.DataFrame({
#     'Name':          ['Mohan','Sohan','Rajeev','Jeevan','Gita','Meenu','Gopal'],
#     'Hours_studied': [2.5,4.0,6.0,8.0,10.0,1.0,5.0],
#     'Marks_obtained':[40,52,64,70,90,10,60]})

# # (i) Student with max marks
# print(df.loc[df['Marks_obtained'].idxmax(),'Name'])


# Company = pd.Series([350,200,800,150],
#     index=['TCS','Reliance','L&T','Wipro'])

# # (i) Companies with profit > 250
# print(Company[Company > 250].index.tolist())



# Score = pd.DataFrame({
#     'Name':['A','B','C','D','E','F'],
#     'Class':[1,2,1,2,2,1],
#     'Score1':[85,74,83,64,77,90],
#     'Score2':[90,86,71,68,62,87],
#     'Score3':[88,80,92,73,72,92]})

# print(Score[['Name','Class']])                        # (i)
# print(Score[Score['Class']==1]['Name']) 