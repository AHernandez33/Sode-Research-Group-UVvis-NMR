import pandas as pd

csv = "np_mrd_1H_1500_expdata.csv"
df = pd.read_csv(csv)

df_unique = df.drop_duplicates(subset=["compound_id"], keep="first")

df_unique.to_csv(
    "np_mrd_1H_1500_expdata_v1.csv",
    index=False
)

print("Saved ML CSV:", len(df_unique))