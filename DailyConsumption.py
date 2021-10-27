import pandas as pd
import matplotlib.pyplot as plt

path_input_day = "new_datas/input_tables/daily/block_0_day.csv"
path_input_hh = "new_datas/input_tables/halfhourly/block_0_halffour.csv"
path_output = "new_datas/output_tables/block_0_mixed.csv"


def daily_consumption_plot(path_output):

    data = pd.read_csv(path_output)
    data.plot(kind="line", figsize=(8, 9) , x=3, y=1)
    # House id = 0 , energy_sum = 1 ,tstp=2 , month =3 ,day_of_month=4 ,time=5,year=6
    plt.ylabel("Energy Consumption (KiloWatt*Hour)")
    plt.xlabel("Days")
    plt.title("Weekly Consumption of a HouseHold")
    plt.show()



