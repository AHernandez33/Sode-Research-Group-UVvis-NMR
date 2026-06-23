# parse_nmrshiftdb2_shift_counts() counts how many times each chemical
# shift appears across the assigned shift records.
#
# Example - 
# 1.23 ppm assigned to 2 atoms -> intensity = 2
# 7.15 ppm assigned to 1 atom  -> intensity = 1


# Peak center = chemical shift (ppm)
# Peak height = shift count from NMRShiftDB2 assignments
# Peak width  = 0.025 ppm
#
# The resulting spectrum is normalized to a maximum intensity of 1.
import os
import re
import json
import base64
import time
import requests 
import numpy as np
import pandas as pd
from rdkit import Chem
# from xml.etree import ElementTree as ET


# smiles_csv = r"C:\Users\herna\Sode Labs\smiles_folder\smiles_NP0000001_NP0050000.csv"
csv = "nmrshift_db2_1H_1500_expdata.csv"
file = r"C:\Users\herna\Sode Labs\NMR\nmrshiftdb2withsignals.sd"

# id_col = "NP_MRD_ID"
# smiles_col = "SMILES"
# name_col = "Natural_Products_Name"

max_rows = 1500
xMin = 0
xMax = 12
points = 1000


def make_spectrum(peaks, x_min=0, x_max=12, n=1000, width=0.025):
    x = np.linspace(x_min, x_max, n)
    y = np.zeros_like(x)

    for shift, intensity in peaks:
        y += intensity * np.exp(-0.5 * ((x - shift) / width) ** 2)

    if y.max() > 0:
        y = y / y.max()

    return x.tolist(), y.tolist()


def parse_nmrshiftdb2_shift_counts(text):
    shift_counts = {}

    # split into separate signal lines/chunks
    chunks = re.split(r"[;\n]", str(text))

    for chunk in chunks:
        chunk = chunk.strip()

        if not chunk:
            continue

        # skip obvious non-shift information
        if re.search(r"\bJ\b|\bHz\b|\bMHz\b|\bK\b|\bsolvent\b|\btemperature\b", chunk, re.IGNORECASE):
            continue

        # remove common integrations like 1H, 2H, 3H, 6H
        chunk = re.sub(r"\b\d+\s*H\b", " ", chunk, flags=re.IGNORECASE)

        # remove reference peak near 0
        chunk = re.sub(r"\b0\.0+\b", " ", chunk)

        # find decimal ppm values only
        matches = re.findall(r"\d+\.\d+", chunk)

        for m in matches:
            shift = float(m)

            # avoid 0/TMS and only keep normal 1H NMR range
            if 0.2 <= shift <= 12:
                shift = round(shift, 4)
                shift_counts[shift] = shift_counts.get(shift, 0) + 1.0

    return shift_counts

def get_pubchem_name(smiles):
    try:
        url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/property/Title/JSON"

        response = requests.post(
            url, 
            data={"smiles": smiles},
            timeout = 1
        )

        if response.status_code != 200:
            return None

        data = response.json()
        props = data.get("PropertyTable", {}).get("Properties", [])
        if len(props) == 0:
            return None
        name = props[0].get("Title", None)
        if name is None:
            return None
        name = str(name).strip()
        if name == "" or name.lower() == "none":
            return None

        return name
    except:
        return None 

pubchem_name_cache = {}

rows = []
supplier = Chem.SDMolSupplier(file, removeHs=False)

for mol_index, mol in enumerate(supplier):
    if len(rows) >= max_rows:
        break

    if mol is None:
        continue

    try:
        smiles = Chem.MolToSmiles(mol)
    except:
        continue

    props = mol.GetPropsAsDict()
    compound_id = props.get("nmrshiftdb2 ID", None)

    if compound_id is None:
        compound_id = props.get("ID", None)

    if compound_id is None:
        if mol.HasProp("_Name"):
            compound_id = mol.GetProp("_Name")
        else:
            compound_id = f"mol_{mol_index}"

    name_keys = [
        "Name",
        "Names",
        "names",
        "Compound Name",
        "IUPAC Name",
        "Title",
        "Molecule Name",
        "nmrshiftdb2 Name"
    ]

    compound_name = None

    for name_key in name_keys:
        if name_key in props:
            value = str(props[name_key]).strip()
            if value and value.lower() != "none" and value != str(compound_id):
                compound_name = value
                break

    if compound_name is None:
        if mol.HasProp("_Name"):
            value = mol.GetProp("_Name").strip()

            if (
                value
                and value.lower() != "none"
                and value != str(compound_id)
            ):
                compound_name = value

    if compound_name is None:
        if smiles in pubchem_name_cache:
            compound_name = pubchem_name_cache[smiles]
        else:
            compound_name = get_pubchem_name(smiles)
            pubchem_name_cache[smiles] = compound_name
            time.sleep(0.2)

    if compound_name is None or compound_name == "" or compound_name == str(compound_id):
        continue 

    # if compound_name is None or compound_name == "":
    #     compound_name = str(compound_id)

    spectrum_keys = []

    for key in props.keys():
        if "Spectrum 1H" in key:
            spectrum_keys.append(key)

    if len(spectrum_keys) == 0:
        continue

    shift_counts = {}

    for key in spectrum_keys:
        spectrum_text = str(props[key])
        counts = parse_nmrshiftdb2_shift_counts(spectrum_text)

        for shift, count in counts.items():
            shift_counts[shift] = shift_counts.get(shift, 0) + count

    if len(shift_counts) == 0:
        continue

    all_shifts = sorted(shift_counts.keys())

    # synthetic intensities
    peaks = [(shift, intensity) for shift, intensity in shift_counts.items()]

    x, y = make_spectrum(
        peaks,
        x_min=xMin,
        x_max=xMax,
        n=points,
        width=0.025
    )

    rows.append({
        "compound_id": compound_id,
        "name": compound_name,
        "smiles": smiles,
        "peaks_ppm": json.dumps(all_shifts),
        "synthetic_peak_intensities": json.dumps(
            [shift_counts[shift] for shift in all_shifts]
        ),
        "x_ppm": json.dumps(x),
        "y_intensity": json.dumps(y),
        "source_file": os.path.basename(file),
        "source_type": ".sdf",
        "intensity_type": "synthetic_shift_count_intensity"
    })

    print("Collected: ", len(rows), "|", compound_name, "|", compound_id)




# def parse_peak_text(text):
#     peaks = []

#     matches = re.findall(
#         r'<peak[^>]*amplitude="([^"]+)"[^>]*center="([^"]+)"',
#         text
#     )

#     for amplitude, center in matches:
#         try:
#             shift = float(center)
#             intensity = float(amplitude)
#         except:
#             continue

#         if 0 <= shift <= 12:
#             peaks.append((shift, intensity))

#     multiplet_matches = re.findall(
#         r'<multiplet[^>]*center="([^"]+)"',
#         text
#     )

#     for center in multiplet_matches:
#         try:
#             shift = float(center)
#         except:
#             continue

#         if 0 <= shift <= 12:
#             peaks.append((shift, 1.0))

#     unique = {}
#     for shift, intensity in peaks:
#         unique[round(shift, 4)] = max(unique.get(round(shift, 4), 0), intensity)

#     return [(shift, intensity) for shift, intensity in unique.items()]


# def extract_spectrum_data_array(text, x_min=0, x_max=12, n=1000):
#     match = re.search(
#         r'<spectrumDataArray[^>]*byteFormat="([^"]+)"[^>]*>(.*?)</spectrumDataArray>',
#         text,
#         re.DOTALL
#     )

#     if not match:
#         return None, None

#     byte_format = match.group(1).lower()
#     encoded = re.sub(r"\s+", "", match.group(2))

#     try:
#         raw = base64.b64decode(encoded)
#     except:
#         return None, None

#     try:
#         if "complex128" in byte_format:
#             arr = np.frombuffer(raw, dtype=np.complex128)
#             y_raw = np.abs(arr)
#         elif "complex64" in byte_format:
#             arr = np.frombuffer(raw, dtype=np.complex64)
#             y_raw = np.abs(arr)
#         elif "float64" in byte_format:
#             y_raw = np.frombuffer(raw, dtype=np.float64)
#         elif "float32" in byte_format:
#             y_raw = np.frombuffer(raw, dtype=np.float32)
#         else:
#             return None, None
#     except:
#         return None, None

#     if len(y_raw) < 2:
#         return None, None

#     x_match = re.search(
#         r'<xAxis[^>]*startValue="([^"]+)"[^>]*endValue="([^"]+)"',
#         text
#     )

#     if not x_match:
#         return None, None

#     try:
#         start_value = float(x_match.group(1))
#         end_value = float(x_match.group(2))
#     except:
#         return None, None

#     x_raw = np.linspace(start_value, end_value, len(y_raw))

#     mask = np.isfinite(x_raw) & np.isfinite(y_raw)
#     x_raw = x_raw[mask]
#     y_raw = y_raw[mask]

#     mask = (x_raw >= x_min) & (x_raw <= x_max)
#     x_raw = x_raw[mask]
#     y_raw = y_raw[mask]

#     if len(x_raw) < 2:
#         return None, None

#     order = np.argsort(x_raw)
#     x_raw = x_raw[order]
#     y_raw = y_raw[order]

#     x_new = np.linspace(x_min, x_max, n)
#     y_new = np.interp(x_new, x_raw, y_raw, left=0, right=0)

#     if np.max(np.abs(y_new)) > 0:
#         y_new = y_new / np.max(np.abs(y_new))

#     return x_new.tolist(), y_new.tolist()


# def extract_compound_id(filename, text):
#     match = re.search(r"NP\d+", filename, re.IGNORECASE)
#     if match:
#         return match.group(0).upper()

#     match = re.search(r"NP\d+", text, re.IGNORECASE)
#     if match:
#         return match.group(0).upper()

#     return os.path.splitext(filename)[0]


# def extract_compound_name(text):
#     match = re.search(
#         r'<identifier[^>]*accession="NMR:1002000"[^>]*name="([^"]+)"',
#         text
#     )

#     if match:
#         return match.group(1).strip()

#     match = re.search(
#         r'<identifier[^>]*name="([^"]+)"',
#         text
#     )

#     if match:
#         return match.group(1).strip()

#     return None


# def read_text_file(path):
#     encodings = ["utf-8", "latin-1", "cp1252"]

#     for enc in encodings:
#         try:
#             with open(path, "r", encoding=enc, errors="ignore") as f:
#                 return f.read()
#         except:
#             continue

#     return ""


# def load_structure_smiles_lookup(smiles_csv):
#     df = pd.read_csv(smiles_csv)

#     smiles_lookup = {}
#     name_lookup = {}

#     for _, row in df.iterrows():
#         compound_id = str(row[id_col]).strip().upper()

#         smiles = row[smiles_col]
#         name = row[name_col]

#         if not pd.isna(smiles):
#             smiles_lookup[compound_id] = str(smiles).strip()

#         if not pd.isna(name):
#             name_lookup[compound_id] = str(name).strip()

#     return smiles_lookup, name_lookup


# structure_smiles_lookup, structure_name_lookup = load_structure_smiles_lookup(smiles_csv)

# print("Structure SMILES loaded:", len(structure_smiles_lookup))
# print("Structure names loaded:", len(structure_name_lookup))

# rows = []


# for root, dirs, files in os.walk(download_folder):
#     for filename in files:
#         if len(rows) >= max_rows:
#             break

#         if "13C" in filename.upper():
#             continue

#         if "HSQC" in filename.upper():
#             continue

#         path = os.path.join(root, filename)
#         ext = os.path.splitext(filename)[1].lower()

#         if ext not in [".txt", ".csv", ".json", ".xml", ".nmrml"]:
#             continue

#         text = read_text_file(path)

#         compound_id = extract_compound_id(filename, text)
#         compound_name = extract_compound_name(text)

#         structure_name = structure_name_lookup.get(compound_id)
#         if compound_name is None or compound_name.upper().startswith("NP"):
#             compound_name = structure_name

#         smiles = structure_smiles_lookup.get(compound_id)

#         if compound_name is None:
#             continue

#         if smiles is None:
#             continue

#         peaks = parse_peak_text(text)

#         if len(peaks) >= 1:
#             x, y = make_spectrum(peaks, x_min=xMin, x_max=xMax, n=points)
#             peaks_ppm = json.dumps([p[0] for p in peaks])
#         else:
#             x, y = extract_spectrum_data_array(text, x_min=xMin, x_max=xMax, n=points)
#             peaks_ppm = None

#         if x is None or y is None:
#             continue

#         rows.append({
#             "compound_id": compound_id,
#             "name": compound_name,
#             "smiles": smiles,
#             "peaks_ppm": peaks_ppm,
#             "x_ppm": json.dumps(x),
#             "y_intensity": json.dumps(y),
#             "source_file": filename,
#             "source_type": ext
#         })

#         print("Collected:", len(rows), "|", compound_name, "|", compound_id)

#     if len(rows) >= max_rows:
#         break


df = pd.DataFrame(rows)
df = df[df["smiles"].notna()]
df = df[df["smiles"].astype(str).str.lower() != "none"]
df = df[df["name"].notna()]
df = df[df["name"].astype(str).str.lower() != "none"]

df.to_csv(csv, index=False)

print("Saved CSV:", csv)
print("Rows:", len(df))
print(df.head())