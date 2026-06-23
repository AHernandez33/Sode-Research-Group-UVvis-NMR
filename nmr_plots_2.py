# plot each and every 1H NMR experimental spectrum

import os
import json
import pandas as pd
import matplotlib.pyplot as plt

csv_file = "nmrshift_db2_1H_1500_expdata.csv"
df = pd.read_csv(csv_file)

print("CSV shape:", df.shape)
print(df.head())

plot_folder = "NMR PLOTS EXP 2"
os.makedirs(plot_folder, exist_ok=True)

# loop through every molecule / row
for index, row in df.iterrows():

    try:
        x = json.loads(row["x_ppm"])
        y = json.loads(row["y_intensity"])

        compound_id = str(row["compound_id"])
        compound_name = str(row["name"])

        plt.figure(figsize=(10, 4))
        plt.plot(x, y, color="black", linewidth=0.5)

        plt.gca().invert_xaxis()

        plt.xlabel("ppm")
        plt.ylabel("intensity")
        plt.title(f"1H NMR Spectrum: {compound_id} - {compound_name}")

        plt.minorticks_on()
        plt.tick_params(axis="x", which="major", length=6, width=1)
        plt.tick_params(axis="x", which="minor", length=3, width=0.8)
        plt.tick_params(axis="y", left=False)

        save_path = os.path.join(plot_folder, f"{compound_id}_1h_nmr.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved {index + 1}/{len(df)}: {save_path}")

    except Exception as e:
        print(f"Skipped row {index} because of error: {e}")
        plt.close()

print("Done plotting all spectra.")