
import numpy as np 
import pandas as pd 
import datetime, time
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)
        
df = pd.read_csv("../input/smart-meters-in-london/halfhourly_dataset/halfhourly_dataset/block_0.csv")        
householdInfos = pd.read_csv("../input/smart-meters-in-london/halfhourly_dataset/halfhourly_dataset/block_0.csv")        

household = "MAC000002"

df = df[df["LCLid"] == household ]

"""
for i in range(df.shape[0]):
    df.loc[i,'datetime'] = datetime.datetime.strptime(df.loc[i,'tstp'].replace('.0000000', ''), '%Y-%m-%d %H:%M:%S')
    df.loc[i,'date'] = df.loc[i,'datetime'].date()
    df.loc[i,'month'] = df.loc[i,'datetime'].strftime("%B")
    df.loc[i,'day_of_month'] = df.loc[i,'datetime'].strftime("%d")
    df.loc[i,'time'] = df.loc[i,'datetime'].strftime('%X')
    df.loc[i,'weekday'] = df.loc[i,'datetime'].strftime('%A')
"""
def myfunc(df):
    for i in range(df.shape[0]):
#    for i in range(40):
        df.loc[i,'datetime'] = datetime.datetime.strptime(df.loc[i,'tstp'].replace('.0000000', ''), '%Y-%m-%d %H:%M:%S')
        df.loc[i,'month'] = df.loc[i,'datetime'].strftime("%B")
        df.loc[i,'day_of_month'] = int(df.loc[i,'datetime'].strftime("%d"))
        df.loc[i,'time'] = df.loc[i,'datetime'].strftime('%X')
        df.loc[i,'year'] = df.loc[i,'datetime'].strftime("%Y")

myfunc(df)



def getPlotOfTheDay(year,month,day,df):
    print(df[(df['year']==year)&(df['month']==month) & (df['day_of_month'] ==float(day) )])
    return df[(df['year']==year)&(df['month']==month) & (df['day_of_month'] ==float(day) )]

df = df[df["energy(kWh/hh)"] != "Null"]
df.loc[:,"energy(kWh/hh)"] = df["energy(kWh/hh)"].astype("float64")


for i in [22,23,24]:
    day = getPlotOfTheDay("2013","November",i,df)
    day.plot(y="energy(kWh/hh)",x="time")