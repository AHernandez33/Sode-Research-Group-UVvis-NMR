# plot each and every 1h nmr (experimental)
import os 
import json
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

csv_file = "nmrshiftdb_1H_1000.csv"
df = pd.read_csv(csv_file)

print(df.shape)
df.head()

plot_folder = "NMR PLOTS"
os.makedirs(plot_folder, exist_ok=True)

row = df.iloc[2]

x = json.loads(row["x_ppm"])
y = json.loads(row["y_intensity"])

compound_id = str(row["compound_id"])

plt.figure(figsize=(10, 4))
plt.plot(x, y, color="black", linewidth=0.5)
plt.gca().invert_xaxis()
plt.xlabel("ppm")
plt.ylabel("intensity")
plt.title(f"1H NMR Spectrum: {row['compound_id']}")

plt.minorticks_on()
plt.tick_params(axis="x", which="major", length=6, width=1)
plt.tick_params(axis="x", which="minor", length=3, width=0.8)
plt.tick_params(axis="y", left=False)

save_path = os.path.join(plot_folder, f"{compound_id}_1h_nmr.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()
print("Saved: ", save_path)