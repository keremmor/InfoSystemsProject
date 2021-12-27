import math
from random import gauss
import random
import ProjectLibrary
import pandas as pd

for i in range(12, 22):
    i = str(i)
    InputFileName = "block_" + i + ".csv"
    OutputFileName = "test" + i + ".csv"
    df_daily_path = "input-tables/daily_dataset/" + InputFileName
    df_hf_hourly_path = "input-tables/halfhourly_dataset/" + InputFileName
    df_output_path = "output-tables/dfOut/" + OutputFileName
    df_processed_path = "output-tables/dfOut/" + OutputFileName

    path_knn = "output-tables/dfAfterKNN/DataFrameKNN_block_" + i + ".csv"

    start_date_daily = '2012-11-01'
    end_date_daily = '2012-11-30'

    df = ProjectLibrary.merge_df_for_knn(df_daily_path, df_hf_hourly_path, df_output_path,
                                         df_processed_path, start_date_daily, end_date_daily)

    ProjectLibrary.write_out_the_given_dataframe(df, path_knn)

# df_daily_path = "input-tables/daily_dataset/" + InputFileName
# df_hf_hourly_path = "input-tables/halfhourly_dataset/" + InputFileName
# df_output_path = "output-tables/dfOut/" + OutputFileName
# df_processed_path = "output-tables/dfOut/" + OutputFileName
#
# path_knn = "output-tables/dfAfterKNN/DataFrameKNN_block_2.csv"
#
# start_date_daily = '2012-11-01'
# end_date_daily = '2012-11-30'
#
# df = ProjectLibrary.merge_df_for_knn(df_daily_path, df_hf_hourly_path, df_output_path,
#                                      df_processed_path, start_date_daily, end_date_daily)
#
# ProjectLibrary.write_out_the_given_dataframe(df, path_knn)
#
#
#
#
#
# InputFileName = "block_3.csv"
# OutputFileName = "test3.csv"
#
# df_daily_path = "input-tables/daily_dataset/" + InputFileName
# df_hf_hourly_path = "input-tables/halfhourly_dataset/" + InputFileName
# df_output_path = "output-tables/dfOut/" + OutputFileName
# df_processed_path = "output-tables/dfOut/" + OutputFileName
#
# path_knn = "output-tables/dfAfterKNN/DataFrameKNN_block_3.csv"
#
# start_date_daily = '2012-11-01'
# end_date_daily = '2012-11-30'
#
# df = ProjectLibrary.merge_df_for_knn(df_daily_path, df_hf_hourly_path, df_output_path,
#                                      df_processed_path, start_date_daily, end_date_daily)
#
# ProjectLibrary.write_out_the_given_dataframe(df, path_knn)
#
#
#
#
#
# InputFileName = "block_4.csv"
# OutputFileName = "test4.csv"
#
# df_daily_path = "input-tables/daily_dataset/" + InputFileName
# df_hf_hourly_path = "input-tables/halfhourly_dataset/" + InputFileName
# df_output_path = "output-tables/dfOut/" + OutputFileName
# df_processed_path = "output-tables/dfOut/" + OutputFileName
#
# path_knn = "output-tables/dfAfterKNN/DataFrameKNN_block_4.csv"
#
# start_date_daily = '2012-11-01'
# end_date_daily = '2012-11-30'
#
# df = ProjectLibrary.merge_df_for_knn(df_daily_path, df_hf_hourly_path, df_output_path,
#                                      df_processed_path, start_date_daily, end_date_daily)
#
# ProjectLibrary.write_out_the_given_dataframe(df, path_knn)
#
#
#
#
#
# InputFileName = "block_0.csv"
# OutputFileName = "test0.csv"
#
# df_daily_path = "input-tables/daily_dataset/" + InputFileName
# df_hf_hourly_path = "input-tables/halfhourly_dataset/" + InputFileName
# df_output_path = "output-tables/dfOut/" + OutputFileName
# df_processed_path = "output-tables/dfOut/" + OutputFileName
#
# path_knn = "output-tables/dfAfterKNN/DataFrameKNN_block_0.csv"
#
# start_date_daily = '2012-11-01'
# end_date_daily = '2012-11-30'
#
# df = ProjectLibrary.merge_df_for_knn(df_daily_path, df_hf_hourly_path, df_output_path,
#                                      df_processed_path, start_date_daily, end_date_daily)
#
# ProjectLibrary.write_out_the_given_dataframe(df, path_knn)
#
#
#
#
#
# InputFileName = "block_1.csv"
# OutputFileName = "test1.csv"
#
# df_daily_path = "input-tables/daily_dataset/" + InputFileName
# df_hf_hourly_path = "input-tables/halfhourly_dataset/" + InputFileName
# df_output_path = "output-tables/dfOut/" + OutputFileName
# df_processed_path = "output-tables/dfOut/" + OutputFileName
#
# path_knn = "output-tables/dfAfterKNN/DataFrameKNN_block_1.csv"
#
# start_date_daily = '2012-11-01'
# end_date_daily = '2012-11-30'
#
# df = ProjectLibrary.merge_df_for_knn(df_daily_path, df_hf_hourly_path, df_output_path,
#                                      df_processed_path, start_date_daily, end_date_daily)
#
# ProjectLibrary.write_out_the_given_dataframe(df, path_knn)
#
#
#
#
#
# InputFileName = "block_5.csv"
# OutputFileName = "test5.csv"
#
# df_daily_path = "input-tables/daily_dataset/" + InputFileName
# df_hf_hourly_path = "input-tables/halfhourly_dataset/" + InputFileName
# df_output_path = "output-tables/dfOut/" + OutputFileName
# df_processed_path = "output-tables/dfOut/" + OutputFileName
#
# path_knn = "output-tables/dfAfterKNN/DataFrameKNN_block_5.csv"
#
# start_date_daily = '2012-11-01'
# end_date_daily = '2012-11-30'
#
# df = ProjectLibrary.merge_df_for_knn(df_daily_path, df_hf_hourly_path, df_output_path,
#                                      df_processed_path, start_date_daily, end_date_daily)
#
# ProjectLibrary.write_out_the_given_dataframe(df, path_knn)
#
#
#
#
# InputFileName = "block_6.csv"
# OutputFileName = "test6.csv"
#
# df_daily_path = "input-tables/daily_dataset/" + InputFileName
# df_hf_hourly_path = "input-tables/halfhourly_dataset/" + InputFileName
# df_output_path = "output-tables/dfOut/" + OutputFileName
# df_processed_path = "output-tables/dfOut/" + OutputFileName
#
# path_knn = "output-tables/dfAfterKNN/DataFrameKNN_block_6.csv"
#
# start_date_daily = '2012-11-01'
# end_date_daily = '2012-11-30'
#
# df = ProjectLibrary.merge_df_for_knn(df_daily_path, df_hf_hourly_path, df_output_path,
#                                      df_processed_path, start_date_daily, end_date_daily)
#
# ProjectLibrary.write_out_the_given_dataframe(df, path_knn)
#
#
# InputFileName = "block_7.csv"
# OutputFileName = "test7.csv"
#
# df_daily_path = "input-tables/daily_dataset/" + InputFileName
# df_hf_hourly_path = "input-tables/halfhourly_dataset/" + InputFileName
# df_output_path = "output-tables/dfOut/" + OutputFileName
# df_processed_path = "output-tables/dfOut/" + OutputFileName
#
# path_knn = "output-tables/dfAfterKNN/DataFrameKNN_block_7.csv"
#
# start_date_daily = '2012-11-01'
# end_date_daily = '2012-11-30'
#
# df = ProjectLibrary.merge_df_for_knn(df_daily_path, df_hf_hourly_path, df_output_path,
#                                      df_processed_path, start_date_daily, end_date_daily)
#
# ProjectLibrary.write_out_the_given_dataframe(df, path_knn)
#
#
# InputFileName = "block_8.csv"
# OutputFileName = "test8.csv"
#
# df_daily_path = "input-tables/daily_dataset/" + InputFileName
# df_hf_hourly_path = "input-tables/halfhourly_dataset/" + InputFileName
# df_output_path = "output-tables/dfOut/" + OutputFileName
# df_processed_path = "output-tables/dfOut/" + OutputFileName
#
# path_knn = "output-tables/dfAfterKNN/DataFrameKNN_block_8.csv"
#
# start_date_daily = '2012-11-01'
# end_date_daily = '2012-11-30'
#
# df = ProjectLibrary.merge_df_for_knn(df_daily_path, df_hf_hourly_path, df_output_path,
#                                      df_processed_path, start_date_daily, end_date_daily)
#
# ProjectLibrary.write_out_the_given_dataframe(df, path_knn)
#
#
# InputFileName = "block_9.csv"
# OutputFileName = "test9.csv"
#
# df_daily_path = "input-tables/daily_dataset/" + InputFileName
# df_hf_hourly_path = "input-tables/halfhourly_dataset/" + InputFileName
# df_output_path = "output-tables/dfOut/" + OutputFileName
# df_processed_path = "output-tables/dfOut/" + OutputFileName
#
# path_knn = "output-tables/dfAfterKNN/DataFrameKNN_block_9.csv"
#
# start_date_daily = '2012-11-01'
# end_date_daily = '2012-11-30'
#
# df = ProjectLibrary.merge_df_for_knn(df_daily_path, df_hf_hourly_path, df_output_path,
#                                      df_processed_path, start_date_daily, end_date_daily)
#
# ProjectLibrary.write_out_the_given_dataframe(df, path_knn)
#
#
#
#
# InputFileName = "block_10.csv"
# OutputFileName = "test10.csv"
#
# df_daily_path = "input-tables/daily_dataset/" + InputFileName
# df_hf_hourly_path = "input-tables/halfhourly_dataset/" + InputFileName
# df_output_path = "output-tables/dfOut/" + OutputFileName
# df_processed_path = "output-tables/dfOut/" + OutputFileName
#
# path_knn = "output-tables/dfAfterKNN/DataFrameKNN_block_10.csv"
#
# start_date_daily = '2012-11-01'
# end_date_daily = '2012-11-30'
#
# df = ProjectLibrary.merge_df_for_knn(df_daily_path, df_hf_hourly_path, df_output_path,
#                                      df_processed_path, start_date_daily, end_date_daily)
#
# ProjectLibrary.write_out_the_given_dataframe(df, path_knn)
