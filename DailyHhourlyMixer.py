import pandas as pd
import datetime, time
import numpy as np


def convert_to_mixed_table(path_input_day, path_input_hh, path_output, first_n_rows):
    df_output = pd.read_csv(path_output, header=0)
    df_day = pd.read_csv(path_input_day, skiprows=range(1, 82), nrows=30)
    df_hh = pd.read_csv(path_input_hh, skiprows=range(1, 3839), nrows=1487)
    df_day = df_day.reindex(np.arange(1487), fill_value=0)
    for i in range(0, first_n_rows):
        df_output.loc[i, 'home_id'] = df_day.loc[i, 'LCLid']
        df_output.loc[i, 'day'] = df_day.loc[i, 'day']
        df_output.loc[i, 'daily_energy_sum'] = float("{0:.6f}".format(df_day.loc[i, 'energy_sum']))
        df_output.loc[i, 'datetime'] = datetime.datetime.strptime(df_hh.loc[i, 'tstp'].replace('.0000000', ''),
                                                                  '%Y-%m-%d %H:%M:%S')
        df_output.loc[i, 'day_month'] = df_output.loc[i, 'datetime'].strftime('%m-%d')
        df_output.loc[i, 'month'] = df_output.loc[i, 'datetime'].strftime("%B")
        df_output.loc[i, 'year'] = df_output.loc[i, 'datetime'].strftime("%Y")
        df_output.loc[i, 'time'] = df_output.loc[i, 'datetime'].strftime('%X')
        df_output.loc[i, 'halfhourly_energy_sum'] = df_hh.loc[i, 'energy']

    df_output.to_csv(path_output, index=False, header=True)
    return path_output


path_input_day = "new_datas/input_tables/daily/block_0_day.csv"
path_input_hh = "new_datas/input_tables/halfhourly/block_0_halffour.csv"
path_output = "new_datas/output_tables/block_0_mixed.csv"
convert_to_mixed_table(path_input_day, path_input_hh, path_output, 1487)
