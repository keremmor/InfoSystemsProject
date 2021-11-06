import pandas as pd
import datetime, time
import numpy as np


def convert(path_input_day, path_output, household, start_date, end_date):
    df_day = pd.read_csv(path_input_day)
    df_day = df_day[df_day["LCLid"] == household]
    df_day_mask = df_day[df_day.day.between(start_date, end_date)]
    df_day_mask.to_csv(path_output, index=False, header=True, float_format='%.4f')


path_input_day = "new_datas/input_tables/daily/block_0_day.csv"
path_output = "new_datas/output_tables/block_0_day_converted.csv"

start_date_daily = '2014-01-01'
end_date_daily = '2014-01-31'

convert(path_input_day, path_output, 'MAC000002', start_date_daily, end_date_daily)
