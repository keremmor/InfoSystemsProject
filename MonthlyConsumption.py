import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as dates

path_input_day = "new_datas/input_tables/daily/block_0_day.csv"
path_input_hh = "new_datas/input_tables/halfhourly/block_0_halffour.csv"
path_output = "new_datas/output_tables/block_0_mixed.csv"


def daily_consumption_plot(path_output,start_date,end_date):

    data = pd.read_csv(path_output)
    df = pd.DataFrame(data, columns=["day", "daily_energy_sum"])
    mask = (df['day'] >= start_date) & (df['day'] <= end_date)
    df = df.loc[mask]
    df.plot(x="day", y=["daily_energy_sum"], kind="line", figsize=(8, 9))
    plt.gca().xaxis.set_major_formatter(dates.DateFormatter('%d\n%m'))
    plt.ylabel("Energy Consumption (KiloWatt*Hour)")
    plt.xlabel("Days")
    plt.title("Monthly Consumption of a HouseHold")
    plt.show()

start_date = '2013-01-01'
end_date = '2013-01-31'
daily_consumption_plot(path_output,start_date,end_date)



