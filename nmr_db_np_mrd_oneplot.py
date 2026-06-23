# plot only one in the NP MRD data csv 
# now loop throughout 1500 molecules

import json
import pandas as pd
import matplotlib.pyplot as plt 
import os

file = "np_mrd_1H_1500_expdata.csv"

output_folder = "NMR PLOTS EXP"
os.makedirs(output_folder, exist_ok=True)
df = pd.read_csv(file)

for i, row in df.iterrows():
    try:
        x = json.loads(row["x_ppm"])
        y = json.loads(row["y_intensity"])

        compound_id = str(row["compound_id"])



        # print("Compound_ID: ", row["compound_id"])
        # print("Name: ", row["name"])
        # print("SMILES: ", row["smiles"])

        plt.figure(figsize=(10, 4))
        plt.plot(x, y, color="black", linewidth=0.5)
        plt.xlabel("Chemical shift (ppm)")
        plt.ylabel("Intensity")
        plt.title(f"{compound_id} - {row['name']}")
        plt.gca().invert_xaxis()

        save_path = os.path.join(
            output_folder,
            f"{compound_id}.png"
        )
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()
        if (i + 1) % 1 == 0:
            print("Saved: ", i + 1)

        
    except Exception as e:
        print(
            "Failed: ", row["compound_id"], e
        )

print("Finished")
print("Plots saved to - ", output_folder)