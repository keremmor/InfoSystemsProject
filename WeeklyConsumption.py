import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv (r"C:\Users\KEREM\Downloads\archive\daily_dataset\daily_dataset\block_0.csv",nrows=7)
df = pd.DataFrame(data, columns=["day", "energy_sum"])
df.plot(x="day", y=[ "energy_sum"], kind="bar", figsize=(8,9))
plt.ylabel("Energy Consumption (KiloWatt*Hour)")
plt.xlabel("Days")
plt.title("Weekly Consumption of a HouseHold")
plt.show()