import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import neighbors, datasets
import numpy as np
from matplotlib.colors import ListedColormap


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
    index_ = 0
    for i in df.index.values:
        if float(df.loc[i, 'energy(kWh/hh)']) > max_value:
            max_value = float(df.loc[i, 'energy(kWh/hh)'])
            index_ = i

    return df.loc[index_, 'clock']


def take_max_load_time_of_days(household_, df_: pd.DataFrame, start_date_, end_date_):
    """
    It takes a dataframe, a household ID and two different date.
    It returns the most used peak time of a month for the specific household and time interval
    """
    _start_date = datetime.datetime.strptime(start_date_, "%Y-%m-%d")
    _end_date = datetime.datetime.strptime(end_date_, "%Y-%m-%d")
    days = (_end_date - _start_date).days
    max_load_times = []

    for i in range(days):
        print("  %", round((i / days) * 100), '..', end="\r")
        cur_day = _start_date + datetime.timedelta(days=i + 1)
        cur_day_str = cur_day.strftime("%Y-%m-%d")
        day_data = get_specific_time_interval(
            household=household_, start_date=cur_day_str, end_date=cur_day_str, df=df_)
        max_load_time = collect_and_detect_max_load_clock(day_data)
        max_load_times.append(max_load_time)

    # print("Max loadt times: ",max_load_times)

    load_dict = {}
    for i in max_load_times:
        try:
            load_dict[i] += 1
        except:
            load_dict[i] = 1

    return sorted(load_dict, key=load_dict.__getitem__, reverse=True)[0]


def get_consumed_energy_of_day(df_, hID, _date):
    """
    It returns the daily consumed energy of a household
    """
    __df = df_[(df_['LCLid'] == hID) & (df_['day'] == _date)]
    e_sum = __df['energy_sum'].iloc[0]
    return e_sum


def get_average_consumed_energy_of_days(df, hID, __start_date, __end_date):
    """
    It returns the mean of daily consumed energies of a month for specific household
    """
    __df = df[df['LCLid'] == hID]
    __df = __df[__df.day.between(__start_date, __end_date)]
    # print(type(__df[['energy_sum']].mean()))
    return __df[['energy_sum']].mean().iloc[0]


def get_average_consumed_energy_of_days_knn(df, hID, __start_date, __end_date):
    """
    It returns the mean of daily consumed energies of a month for specific household
    """
    __df = df[df['LCLid'] == hID]
    __df = __df[__df.day.between(__start_date, __end_date)]
    # print(type(__df[['energy_sum']].mean()))
    return __df[['energy(kWh/hh)']].mean().iloc[0]


def transform_datetime_to_decimal(dTimeStr):
    hour_ = int(dTimeStr[:2])
    half_ = int(dTimeStr[3:5])
    if half_ > 0:
        hour_ = hour_ + 1 / 2
    return hour_


def plot_labeled_data(transformed_data, labels_):
    """
       It plots the labeled data with 5 classes
       """
    for i in range(len(transformed_data)):
        if labels_[i] == 1:
            plt.plot(transformed_data[i][0], transformed_data[i][1], 'bo')
        elif labels_[i] == 0:
            plt.plot(transformed_data[i][0], transformed_data[i][1], 'ro')
        elif labels_[i] == 2:
            plt.plot(transformed_data[i][0], transformed_data[i][1], 'co')
        elif labels_[i] == 3:
            plt.plot(transformed_data[i][0], transformed_data[i][1], 'mo')

        else:
            plt.plot(transformed_data[i][0], transformed_data[i][1], 'yo')

    plt.xlabel('Peak Load Time')
    plt.ylabel('Total consumed Energy (kW)')

    plt.show()


def predict_class_of_new_household(transformed_data_list, label_list, peak_time, energy):
    """
    This function takes input transformed data list and label list which come from Kmeans.py as output.
    And also takes average monthly energy of a house and peak time of household and makes a predicition about the class of new added household.
    It returns the predicted class of a household
    """
    df1 = pd.DataFrame(label_list,
                       columns=['label'])

    df2 = pd.DataFrame(transformed_data_list,
                       columns=['peak_hour', 'average_energy'])

    df_result = pd.concat([df2, df1], axis=1, join='inner')
    X = df_result.iloc[:, :-1].values
    y = df_result.iloc[:, 2].values
    h = .02

    n_neighbors = 6
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    clf.fit(X, y)

    dataClass = clf.predict([[peak_time, energy]])

    if dataClass == 0:
        print('Prediction: ', dataClass)
        return dataClass
    elif dataClass == 1:
        print('Prediction: ', dataClass)
        return dataClass
    elif dataClass == 2:
        print('Prediction: ', dataClass)
        return dataClass
    elif dataClass == 3:
        print('Prediction: ', dataClass)
        return dataClass

    else:
        print('Prediction: ', dataClass)
        return dataClass


def df_merge_knn(df1: pd.DataFrame, df2: pd.DataFrame, label_list, transformed_data_list):
    """
        This function takes 2 dataframes and 2 list as input.It concatenates the df1 to df2.
        This function is specific for KNN.py
    """

    df_result = pd.concat([df2, df1], axis=1, join='inner')
    return df_result


def append_new_item_to_lists(transformed_data_list, label_list, prediction_of_hhold, peak_time, energy):
    """
       It appends the peak time and energy of new added household to the new list.
       It is made for K-NearestNeighbour.py
       It returns the 2 lists
       """
    label_list.extend(prediction_of_hhold)
    transformed_data_list.append([peak_time, energy])

    return label_list, transformed_data_list


def knn_classification(df: pd.DataFrame):
    """
       This function makes the KNN classification
       """
    n_neighbors = 6
    X = df.iloc[:, :-1].values
    y = df.iloc[:, 2].values
    h = .02

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF', "#ad40ff", "#ffff00"])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#00AAFF', "#ad40ff", "#ffff00"])

    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
    clf.fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure()

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)

    plt.title("Classification of new added household using KNN (k = %i)" % (n_neighbors))
    plt.show()
