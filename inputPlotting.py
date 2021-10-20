import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv (r"C:\Users\KEREM\Downloads\archive\daily_dataset\daily_dataset\block_0.csv")
data = data.head()
df = pd.DataFrame(data, columns=["day", "energy_median", "energy_std"])
df.plot(x="day", y=["energy_median", "energy_std"], kind="line", figsize=(9,8))
plt.ylabel("Energy Consumption (KiloWatt*Hour)")
plt.title("Energy Consumption Comparison Between Two Variables")
plt.show()
