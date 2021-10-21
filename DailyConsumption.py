import pandas as pd
import matplotlib.pyplot as plt


def logic(index):
    if index > 40:
        return True

    return False


data = pd.read_csv (r"C:\Users\KEREM\Downloads\archive\halfhourly_dataset\halfhourly_dataset\block_0.csv",skiprows = lambda x: logic(x))
df = pd.DataFrame(data, columns=["tstp", "energy"])
df.plot(x="tstp", y=[ "energy"], kind="line", figsize=(8,9))
plt.ylabel("Energy Consumption (KiloWatt*Hour)")
plt.xlabel("Days")
plt.title("Weekly Consumption of a HouseHold")
plt.show()