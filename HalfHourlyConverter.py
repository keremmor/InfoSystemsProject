import pandas as pd
import datetime, time
import numpy as np


def convert(path_input_hh, path_output, household, start_date, end_date):
    df_hh = pd.read_csv(path_input_hh)
    df_hh = df_hh[df_hh["LCLid"] == household]
    df_hh_mask = df_hh.loc[(df_hh['tstp'] > start_date) & (df_hh['tstp'] < end_date)]
    # df_hh_mask['tstp'] = datetime.datetime.strptime(df_hh_mask.loc['tstp'].replace('.0000000', ''),
    #                                                 '%Y-%m-%d %H:%M:%S')
    df_hh_mask.to_csv(path_output, index=False, header=True, float_format='%.4f')


path_input_day = "new_datas/input_tables/halfhourly/block_0_halffour.csv"
path_output = "new_datas/output_tables/block_0_hh_converted.csv"

start_date_hh = ('2013-01-01 00:00')
end_date_hh = ('2013-01-31 00:00')

convert(path_input_day, path_output, 'MAC000002', start_date_hh, end_date_hh)
