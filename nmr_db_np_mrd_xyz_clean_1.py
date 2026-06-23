import os
import shutil
import pandas as pd

csv = "np_mrd_1H_1500_expdata.csv"
xyz_folder = "XYZ FILES"
clean_folder = "XYZ FILES 1"

os.makedirs(clean_folder, exist_ok=True)

df = pd.read_csv(csv)
df_unique = df.drop_duplicates(subset=["compound_id"], keep="first")

for _, row in df_unique.iterrows():
    compound_id = str(row["compound_id"])

    matches = [
        f for f in os.listdir(xyz_folder)
        if f.startswith(compound_id + "_") and f.endswith(".xyz")
    ]

    if len(matches) > 0:
        src = os.path.join(xyz_folder, matches[0])
        dst = os.path.join(clean_folder, matches[0])
        shutil.copy2(src, dst)
    else:
        print("Missing XYZ for:", compound_id)

print("Unique CSV rows:", len(df_unique))
print("Unique XYZ files copied:", len(os.listdir(clean_folder)))