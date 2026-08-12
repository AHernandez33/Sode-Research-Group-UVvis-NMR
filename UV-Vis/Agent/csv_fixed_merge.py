# Merge both 
import pandas as pd
import numpy as np
from rdkit import Chem
import ast

file_1 = "uv_vis_400_molecules.csv"
file_2 = "UV_w_SMILES.csv"
output = "UV_vis_merged_w_source.csv"

df1 = pd.read_csv(file_1)
df2 = pd.read_csv(file_2)

wl_col = [str(wl) for wl in range(220, 401)]

def convert_list(value):
    if isinstance(value, list):
        return value 
    return ast.literal_eval(str(value))



def canonical(smiles):
    if pd.isna(smiles):
        return None
    
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)

df2 = df2.rename(columns={df2.columns[0]: "smiles"})

if len(df2.columns) - 1 != 181:
    raise ValueError(
        f"The 932-molecule CSV has {len(df2.columns) - 1} absorbance columns, "
        "but 181 were expected."
    )

df2.columns = ["smiles"] + wl_col

df1["wavelength"] = df1["wavelength"].apply(convert_list)
df1["absorbance"] = df1["absorbance"].apply(convert_list)

rows = []

for _, row in df1.iterrows():
    wavelengths = row["wavelength"]
    absorbances = row["absorbance"]

    if len(wavelengths) != 181 or len(absorbances) != 181:
        continue

    new_row = {"smiles": row["smiles"]}

    for wavelength, absorbance in zip(wavelengths, absorbances):
        new_row[str(int(float(wavelength)))] = absorbance

    rows.append(new_row)

df1 = pd.DataFrame(rows)

# Ensure both datasets have exactly the same columns
df1 = df1.reindex(columns=["smiles"] + wl_col)

df1["canonical_smiles"] = df1["smiles"].apply(canonical)
df2["canonical_smiles"] = df2["smiles"].apply(canonical)

df1["dataset_source"] = "NIST"
df2["dataset_source"] = "PNNL"

merged = pd.concat([df2, df1], ignore_index=True)

before = len(merged)

merged = merged[merged["canonical_smiles"].notna()]
merged = merged.drop_duplicates(
    subset="canonical_smiles",
    keep="first"
)

duplicates_removed = before - len(merged)

merged = merged.drop(columns="canonical_smiles")

merged = merged[
    ["smiles", "dataset_source"] + wl_col
]

merged.to_csv(output, index=False)

print("Finished")