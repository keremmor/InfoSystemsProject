from os import write
from typing import Dict
import pandas as pd

import datetime


InputFileName = "block_0.csv"
OutputFileName = "test4.csv"

df_daily_path = "/Users/musabakici/Desktop/Out of software/Lectures/EE4054/Datasets/daily_dataset/"+InputFileName
df_hf_hourly_path = "/Users/musabakici/Desktop/Out of software/Lectures/EE4054/Datasets/halfhourly_dataset/"+InputFileName
df_output_path = "/Users/musabakici/Desktop/Out of software/Lectures/EE4054/Datasets/dfOut/"+OutputFileName
df_processed_path = "/Users/musabakici/Desktop/Out of software/Lectures/EE4054/Datasets/dfOut/"+OutputFileName


#df_ = pd.read_csv(df_hf_hourly_path, header=0)


def get_specific_time_interval(household, start_date, end_date, df):
    """
    It takes an input dataframe and extracts the data with specific household and specific time. 

    """
    df_day = df[df["LCLid"] == household]
    df_day_mask = df_day[df_day.day.between(start_date, end_date)]

    return df_day_mask


def countHouseNumber(df: pd.DataFrame):
    """
    It takes an input dataframe and counts the all unique household number and saves their IDs. 
    This function is specific for half hourly datasets
    """
    
    print("Counting starts...")
    HouseIds = []
    HouseIds.append(df.loc[:0, 'LCLid'].iloc[0])
    i = 0
    while i < df.shape[0]:
        if df.loc[i, 'LCLid'] != HouseIds[-1]:
            HouseIds.append(df.loc[i, 'LCLid'])
        i += 17280

    print("Length : ", len(HouseIds))
    return len(HouseIds), HouseIds


def convert_to_be_with_day_feature(_df: pd.DataFrame):
    """
    It takes an input dataframe and extracts the same data
    with additional two columns which are the 'day' and 'clock'.
    The columns refers only the date(Ex : 2021-03-12) and only the clock(Ex : 04:40:10) time in order.
    
    """
    
    _df['day'] = _df.tstp.str[:10]
    _df['clock'] = _df.tstp.str[11:19]
    _df.drop('tstp', inplace=True, axis=1)
    return _df


def write_out_the_given_dataframe(df_, out_path):
    """
    It takes an input dataframe and write out the given path
    """
    
    df_.to_csv(out_path, index=False,
               header=True, float_format='%.4f')


def read_dataframe(data_path):
    """
    It takes a csv file path to read and return the dataframe
    """
    
    df_ = pd.read_csv(data_path, header=0)
    return df_


def collect_and_detect_max_load_clock(df: pd.DataFrame):

    """
    It takes a dataframe which should include hourly info of one day , 
    and detects the maximum energy consumed time as a peak load.
    """
    max_value = 0
    index_=0
    for i in df.index.values:
        if float(df.loc[i, 'energy(kWh/hh)']) > max_value:
            max_value = float(df.loc[i, 'energy(kWh/hh)'])
            index_=i
    
    return df.loc[index_,'clock']


def take_max_load_time_of_days(household_, df_: pd.DataFrame, start_date_, end_date_):
    
    """
    It takes a dataframe, a household ID and two different date.
    It returns the most used peak time of a month for the specific household and time interval
    """
    _start_date = datetime.datetime.strptime(start_date_, "%Y-%m-%d")
    _end_date = datetime.datetime.strptime(end_date_, "%Y-%m-%d")
    days = (_end_date-_start_date).days
    max_load_times=[]
    
    for i in range(days):
        print("%",round((i/days)*100,2),'...',end="\r")
        cur_day  = _start_date +datetime.timedelta(days=i+1)
        cur_day_str = cur_day.strftime("%Y-%m-%d")
        day_data = get_specific_time_interval(
            household=household_, start_date=cur_day_str, end_date=cur_day_str, df=df_)
        max_load_time= collect_and_detect_max_load_clock(day_data)
        max_load_times.append(max_load_time)
    
    #print("Max loadt times: ",max_load_times)

    load_dict = {}
    for i in max_load_times:
        try:
            load_dict[i]+=1
        except:
            load_dict[i]=1
    
    return sorted(load_dict, key=load_dict.__getitem__, reverse=True)[0]


def get_consumed_energy_of_day(df_,hID,_date):
    """
    It returns the daily consumed energy of a household
    """
    __df = df_[(df_['LCLid']==hID) & (df_['day']==_date)]
    e_sum = __df['energy_sum'].iloc[0]
    return e_sum


def get_average_consumed_energy_of_days(df,hID,__start_date,__end_date):

    """
    It returns the mean of daily consumed energies of a month for specific household
    """
    __df = df[df['LCLid']==hID]
    __df = __df[__df.day.between(__start_date,__end_date)]
    #print(type(__df[['energy_sum']].mean()))
    return __df[['energy_sum']].mean().iloc[0]

start_date_daily = '2012-11-01'
end_date_daily = '2012-11-30'

df__daily = read_dataframe(df_daily_path)

_df_ = read_dataframe(df_processed_path)

LenOfHouseIds, HouseIds = countHouseNumber(_df_)

for i in HouseIds:
    try:
        t_time = take_max_load_time_of_days(i,_df_,start_date_daily,end_date_daily)
        e_sum = get_average_consumed_energy_of_days(df__daily,i,start_date_daily,end_date_daily)
        print('Max load time of ',i," is ",t_time," and avrg consumed energy is:", e_sum)
    except:
        print('The data does not exist for ',i," between given interval ",start_date_daily," : ",end_date_daily)





