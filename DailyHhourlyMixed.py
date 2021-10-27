import pandas as pd
import datetime, time


def convert_to_mixed_table(path_input_day, path_input_hh, path_output, first_n_rows):
    first_n_rows = first_n_rows + 1
    df_output = pd.read_csv(path_output)
    df_day = pd.read_csv(path_input_day)
    df_hh = pd.read_csv(path_input_hh)

    for i in range(0, first_n_rows):
        df_output.loc[i, 'a'] = df_day.loc[i, 'LCLid']
        df_output.loc[i, 'b'] = float("{0:.6f}".format(df_day.loc[i, 'energy_sum']))
        df_output.loc[i, 'datetime'] = datetime.datetime.strptime(df_hh.loc[i, 'tstp'].replace('.0000000', ''),
                                                                  '%Y-%m-%d %H:%M:%S')
        df_output.loc[i, 'month'] = df_output.loc[i, 'datetime'].strftime("%B")
        df_output.loc[i, 'day_of_month'] = int(df_output.loc[i, 'datetime'].strftime("%d"))
        df_output.loc[i, 'time'] = df_output.loc[i, 'datetime'].strftime('%X')
        df_output.loc[i, 'year'] = df_output.loc[i, 'datetime'].strftime("%Y")

    df_output.to_csv(path_output, index=False, header=False)

path_input_day = "new_datas/input_tables/daily/block_0_day.csv"
path_input_hh = "new_datas/input_tables/halfhourly/block_0_halffour.csv"
path_output = "new_datas/output_tables/block_0_mixed.csv"

convert_to_mixed_table(path_input_day, path_input_hh, path_output, first_n_rows=48)
return path_output
