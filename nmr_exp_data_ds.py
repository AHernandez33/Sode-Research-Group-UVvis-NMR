# draft 1 for NMR web scrapiing data 

# import everything 
import re
import os
import time
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import jcamp 
from jcamp import jcamp_readfile
import rdkit 
from rdkit import Chem 
import json

# database files
# jcamp_folder = "nmrshiftdb_jcamp_files"
structure_file = r"C:\Users\herna\Sode Labs\NMR\nmrshiftdb2withsignals.sd"

def make_spectrum(peaks, x_min=0, x_max=12, n=2000, width=0.025):    
    x = np.linspace(x_min, x_max, n)
    y = np.zeros_like(x)

    for shift, intensity in peaks:
        y += intensity * np.exp(-0.5 * ((x - shift) / width) ** 2)

    if y.max() > 0:
        y = y / y.max()

    return x.tolist(), y.tolist()


def parse_nmrshiftdb_1h_spectrum(spectrum_text):
    peaks = []
    parts = spectrum_text.split("|")
    for part in parts:
        if not part.strip():
            continue

        values = part.split(";")
        try:
            shift = float(values[0])
        except:
            continue 

        if 0 <= shift <= 12:
            peaks.append((shift, 1.0))

    return peaks 
rows = []

# def resample_spectrum(x, y, x_min=0, x_max=12, n=300):
#     """
#     this takes x and y spectrum data and scale it down to 300 points
#     """
#     x = np.array(x, dtype=float)
#     y = np.array(y, dtype=float)
#     mask = np.isfinite(x) & np.isfinite(y)
#     x = x[mask]
#     y = y[mask]

#     if len(x) < 2:
#         return None, None

#     order = np.argsort(x)
#     x = x[order]
#     y = y[order]

#     mask = (x >= x_min) & (x <= x_max)
#     x = x[mask]
#     y = y[mask]

#     if len(x) < 2: 
#         return None, None 

#     x_new = np.linspace(x_min, x_max, n)
#     f = interp1d(x, y, bounds_error=False, fill_value=0)
#     y_new = f(x_new)

#     if np.max(np.abs(y_new)) > 0:
#         y_new = y_new / np.max(np.abs(y_new))

#     return x_new.tolist(), y_new.tolist()

# def load_structures_from_sdf(sdf_file):
#     """
#     this reads the NMRShiftDB structures from the sdf file which converts the molecules to SMILES strings
#     """

supplier = Chem.SDMolSupplier(
    structure_file,
    sanitize=False,
    removeHs=False
)

for mol in supplier:
    if mol is None:
        continue

    if len(rows) >= 1000:
        break

    try:
        smiles = Chem.MolToSmiles(mol)
    except:
        continue

    props = mol.GetPropsAsDict()
    compound_id = props.get("nmrshiftdb2 ID", None)

    for key, value in props.items():
        if not key.startswith("Spectrum 1H"):
            continue

        peaks = parse_nmrshiftdb_1h_spectrum(value)

        if len(peaks) < 2:
            continue

        x, y = make_spectrum(peaks, n=2000)

        rows.append({
            "compound_id": compound_id,
            "spectrum_key": key,
            "smiles": smiles,
            "formula": props.get("FORMULA", None),
            "solvent": props.get("Solvent", None),
            "temperature_K": props.get("Temperature [K]", None),
            "field_strength_MHz": props.get("Field Strength [MHz]", None),
            "peaks_ppm": json.dumps([p[0] for p in peaks]),
            "x_ppm": json.dumps(x),
            "y_intensity": json.dumps(y)
        })

        break

    if len(rows) % 100 == 0 and len(rows) > 0:
        print("Collected:", len(rows))

    # structures = {}
    # supplier = Chem.SDMolSupplier(
    #     structure_file,
    #     sanitize=False,
    #     removeHs=False
    # )
    # # count = 0

    # for mol in supplier:
    #     if mol is None:
    #         continue

    #     if len(rows) >= 1000:
    #         break 


    #     # count += 1
    #     # print("\nMOLECULE", count)

    #     try:
    #         smiles = Chem.MolToSmiles(mol)
    #     except:
    #         continue 


    #     smiles = Chem.MolToSmiles(mol)

    #     props = mol.GetPropsAsDict()
        # for key in props:
        #     if key.startswith("Spectrum 1H"):
        #         print("KEY: ", key)
        #         print(props[key][:1000])
        #         raise SystemExit 

        # if count >= 5: 
        #     break 
        # compound_id = None

#         for possible_id in ["nmrshiftdb2 ID", "ID", "NMRSHIFTDB_ID", "nmrshiftdb_id", "DATABASE_ID"]:
#             if possible_id in props:
#                 compound_id = str(props[possible_id])
#                 break

#         # fallback - use the molecule name 
#         if compound_id is None:
#             compound_id = mol.GetProp("_Name") if mol.HasProp("_Name") else None
#         if compound_id is None:
#             continue
#         structures[compound_id] = {
#             "smiles": smiles,
#             "name": mol.GetProp("_Name") if mol.HasProp("_Name") else None,
#             "formula": props.get("FORMULA", None),
#             "props": props 
#         }

#         if len(structures) >= 5:
#             break

#     return structures

# # load structures and smiles from the sdf file 
# structures = load_structures_from_sdf(structure_file)
# print("Structures loaded: ", len(structures))
# for compound_id, info in structures.items():
#     print("\nID: ", compound_id)
#     print("SMILES: ", info["smiles"])
#     print("NAME: ", info["name"])

#     print("PROPERTIES: ")
#     for key, value in info["props"].items():
#         print(key, "=", str(value)[:500])
# rows = []
    
# # this loops through downloaded nmr shfitdb jcamp files 
# for filename in os.listdir(jcamp_folder):
#     if len(rows) >= 1000:
#         break
            
#     if not filename.lower().endswith((".jdx", ".dx", ".jcamp")):
#         continue 
#     path = os.path.join(jcamp_folder, filename)

#     try:
#         data = jcamp_readfile(path)
#     except Exception as e:
#         print("Could not read: ", filename , e)
#         continue
#     if "x" not in data or "y" not in data:
#         continue 

#     x_raw = data["x"]
#     y_raw = data["y"]

#     x_ppm, y_intensity = resample_spectrum(x_raw, y_raw, n=100)

#     if x_ppm is None:
#         continue 
    
#     compound_id = os.path.splitext(filename)[0]
#     info = structures.get(compound_id, {})

#     rows.append({
#         "compound_id": compound_id,
#         "name": info.get("name", None),
#         "formula": info.get("formula", None),
#         "smiles": info.get("smiles", None),
#         "x_ppm": json.dumps(x_ppm),
#         "y_intensity": json.dumps(y_intensity),
#         "source_file": filename
#     })

#     if len(rows) % 100 == 0:
#         print("Collected: ", len(rows))

    # compound_id = props.get("nmrshiftdb2 ID", None)


    # for key, value in props.items():

    #     if not key.startswith("Spectrum 1H"):
    #         continue

    #     peaks = parse_nmrshiftdb_1h_spectrum(value)

    #     if len(peaks) < 2:
    #         continue

    #     x, y = make_spectrum(peaks, n=300)

    #     rows.append({
    #         "compound_id": compound_id,
    #         "spectrum_key": key,
    #         "smiles": smiles,
    #         "formula": props.get("FORMULA", None),
    #         "solvent": props.get("Solvent", None),
    #         "temperature_K": props.get("Temperature [K]", None),
    #         "field_strength_MHz": props.get("Field Strength [MHz]", None),

    #         "peaks_ppm": json.dumps([p[0] for p in peaks]),

    #         "x_ppm": json.dumps(x),
    #         "y_intensity": json.dumps(y)
    #     })

    #     break

    # if len(rows) % 100 == 0 and len(rows) > 0:
    #     print("Collected:", len(rows))


df = pd.DataFrame(rows)
df.to_csv("nmrshiftdb_1H_1000.csv", index=False)

# print("Saved CSV")
# print("Rows:", len(df))
# df = pd.DataFrame(rows)
# df.to_csv("1H_nmr_exp.csv", index=False)
# print("Saved CSV")
# print("Rows: ", len(df))

#     return x.tolist(), y.tolist()

# def parse_nmr_peaks(text):
#     """
#     Replace this with the SDBS parser asfter inspecting pages (soon)
#     Output should be  [(shift_ppm, intensity), ...]
#     """

#     peaks = []
#     matches = re.findall(r"(\d+\.\d+)\s*\(([^)]*)\)", text)
#     for shift, info in matches:
#         shift = float(shift)

#         # this is for the crude integration estimate 
#         h_match = re.search(r"(\d+)\s*H", info)
#         intensity = float(h_match.group(1)) if h_match else 1.0
#         peaks.append((shift, intensity))

#     return peaks 
# rows = []
# candidate_ids = range(1, 1000) # this is getting 5000 molecules for a dataset

# for sdbs_id in candidate_ids:
#     if len(rows) >= 1000:
#         break 

#     url = f"https://sdbs.db.aist.go.jp/CompoundView.aspx?sdbsno={sdbs_id}"

#     try:
#         r = requests.get(url, timeout=10)
#     except requests.RequestException:
#         continue 

#     soup = BeautifulSoup(r.text, "html.parser")
#     text = soup.get_text(" ", strip=True)

#     peaks = parse_nmr_peaks(text)
#     if len(peaks) < 2:
#         continue 

#     x, y = make_spectrum(peaks, n=300)

#     rows.append({
#         "sdbs_id" : sdbs_id,
#         "name": None,
#         "formula": None,
#         "cas": None,
#         "smiles": None,
#         "x_ppm": json.dumps(x),
#         "y_intensity": json.dumps(y)
#     })

#     time.sleep(1)
# df = pd.DataFrame(rows)
# df.to_csv("sdbs_1h_nmr_exp.csv", index=False)


