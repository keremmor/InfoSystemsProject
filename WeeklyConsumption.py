import pandas as pd
import matplotlib.pyplot as plt


def weekly_consumption_of_household(path, start_date, end_date):
    data = pd.read_csv(path)
    df = pd.DataFrame(data, columns=["day", "daily_energy_sum"])
    mask = (df['day'] > start_date) & (df['day'] <= end_date)
    df = df.loc[mask]
    df.plot(x="day", y=["daily_energy_sum"], kind="line", figsize=(8, 9))
    plt.ylabel("Energy Consumption (KiloWatt*Hour)")
    plt.xlabel("Days")
    plt.title("Weekly Consumption of a HouseHold")
    plt.show()


path = "new_datas/output_tables/block_0_mixed.csv"
start_date = '2013-01-12'
end_date = '2013-01-19'
weekly_consumption_of_household(path, start_date, end_date)
