import pandas as pd
import datetime
import time
import numpy as np


def countHouseNumber(df: pd.DataFrame):

    print("Counting starts....")

    HouseIds = []
    HouseIds.append(df.loc[:0, 'LCLid'].iloc[0])

    i = 0

    while i < df.shape[0]:
        if df.loc[i, 'LCLid'] != HouseIds[-1]:
            HouseIds.append(df.loc[i, 'LCLid'])
        i += 350

    print("Length : ", len(HouseIds))
    

    return len(HouseIds), HouseIds


def readCsv(path_input_day):
    print("Reading dataset..")
    df_day = pd.read_csv(path_input_day)
    return df_day


def writeOut(df_day_mask, path_output):
    df_day_mask.to_csv(path_output, index=False,
                       header=True, float_format='%.4f')


def convert(household, start_date, end_date, df):
    print("Converting datasets..")
    df_day = df[df["LCLid"] == household]
    df_day_mask = df_day[df_day.day.between(start_date, end_date)]
    print("Converting finished")

    return df_day_mask


path_input_day = "/Users/musabakici/Desktop/Out of software/Lectures/EE4054/Datasets/daily_dataset/block_0.csv"
path_output = "/Users/musabakici/Desktop/Out of software/Lectures/EE4054/Datasets/dfOut/book_0_processed.csv"

start_date_daily = '2014-01-01'
end_date_daily = '2014-01-31'

df = readCsv(path_input_day)
len,HouseIds = countHouseNumber(df)
df_ = convert(HouseIds[0],start_date_daily, end_date_daily, df)
writeOut(df_, path_output)
