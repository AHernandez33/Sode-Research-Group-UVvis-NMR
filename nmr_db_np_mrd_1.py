import os
import re
import json
import numpy as np
import pubchempy as pcp
import pandas as pd
from rdkit import Chem
from xml.etree import ElementTree as ET
import requests
# import pyopsin
import cirpy


download_folder = r"C:\Users\herna\Sode Labs\downloads"
output_csv = "np_mrd_1H_1000_expdata.csv"
# lotus_file = r"C:\Users\herna\Sode Labs\LOTUS_DB.smi"
# lotus_lookup = {}
smiles_csv = r"C:\Users\herna\Sode Labs\smiles_folder\smiles_NP0000001_NP0050000.csv"

id_col = "NP_MRD_ID"
smiles_col = "SMILES"
name_col = "Natural_Products_Name"
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

def parse_peak_text(text):
    """
    Extract 1H NMR peaks from the NP MRD database ----- csv files
    Takes from 0 to 12 ppm peaks and overall data 
    """

    peaks = []

    matches = re.findall(
        r'<peak[^>]*amplitude="([^"]+)"[^>]*center="([^"]+)"',
        text
    )

    for amplitude, center in matches:
        try:
            shift = float(center)
            intensity = float(amplitude)
        except:
            continue

        if 0 <= shift <= 12:
            peaks.append((shift, intensity))

    # this removes duplicate shifts 
    unique = {}
    for shift, intensity in peaks:
        unique[round(shift, 4)] = max(unique.get(round(shift, 4), 0), intensity)

    return [(shift, intensity) for shift, intensity in unique.items()]
def extract_smiles(text):
    """
    Find smiles inside the downloaded files
    """

    patterns = [
        r'"smiles"\s*:\s*([^"]+)"',
        r"<smiles>(.*?)</smiles>",
        r"SMILES\s*[:=]\s*([^\s,;]+)",
        r"smiles\s*[:=]\s*([^\s,;]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None

def extract_smiles_from_structure(text):
    """
    build a smiles string from the atomlist and bondlist 
    """
    try:
        root = ET.fromstring(text)
    except:
        return None

    ns = {"nmr": "http://nmrml.org/schema"}

    atom_dict = {}
    mol = Chem.RWMol()

    # read the atoms 
    for atom in root.findall(".//nmr:atomList/nmr:atom", ns):
        atom_id = atom.attrib.get("id")
        element = atom.attrib.get("elementType")

        if atom_id is None or element is None:
            continue

        if element == "H":
            continue 
            
        try:
            rd_atom = Chem.Atom(element)
            idx = mol.AddAtom(rd_atom)
            atom_dict[atom_id] = idx
        except:
            continue

    # read the bonds
    for bond in root.findall(".//nmr:bondList/nmr:bond", ns):
        refs = bond.attrib.get("atomRefs")
        order = bond.attrib.get("order", "1")

        if refs is None:
            continue

        parts = refs.split()
        if len(parts) != 2:
            continue 

        a1, a2 = parts
        if a1 not in atom_dict or a2 not in atom_dict:
            continue 

        if order == "1":
            bond_type = Chem.BondType.SINGLE

        elif order == "2":
            bond_type = Chem.BondType.DOUBLE
        elif order == "3":
            bond_type = Chem.BondType.TRIPLE
        else:
            bond_type = Chem.BondType.SINGLE
        try:
            mol.AddBond(atom_dict[a1], atom_dict[a2], bond_type)
        except:
            continue

    try:
        final_mol = mol.GetMol()
        Chem.SanitizeMol(final_mol)
        smiles = Chem.MolToSmiles(final_mol)
        return smiles
    except:
        return None 

def extract_smiles_from_pubchem(compound_name):
    """
    uses pubchem as an alt
    """
    if compound_name is None:
        return None

    try:
        results = pcp.get_compounds(compound_name, "name")
        if len(results) == 0:
            return None 

        return results[0].canonical_smiles
    except:
        return None

def extract_smiles_from_cactus(compound_name):
    """
    use the NIH cactus resolver to turn name into a SMILES string
    """

    if compound_name is None:
        return None

    try:
        url_cactus = (
            "https://cactus.nci.nih.gov/chemical/structure/"
            + compound_name +
            "/smiles"
        )

        r = requests.get(url_cactus, timeout=5)

        if r.status_code == 200:
            smiles = r.text.strip()

            if smiles and "Page not found" not in smiles:
                return smiles

    except:
        pass

    return None

# add another function?
# another function (try pyopsin)
# def extract_smiles_from_opsin(compound_name):
#     """
#     this uses opsin to extract smiles from name 
#     """
#     if compound_name is None:
#         return None

#     try:
#         smiles = pyopsin.name_to_smiles(compound_name)
#         if smiles is not None and smiles.strip() != "":
#             return smiles
#     except:
#         pass
#     return None

def load_lotus_lookup(lotus_file):
    lookup = {}
    try:
        with open(lotus_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) < 1:
                    continue
                smiles = parts[0]
                name = " ".join(parts[1:])
                lookup[name.lower()] = smiles
    except Exception as e:
        print("Can't load lotus: ", e)

    return lookup

def extract_compound_name(text):
    """ 
    This extracts the compound name from the nmrML file 
    """
    match = re.search(
        r'<identifier[^>]*accession="NMR:1002000"[^>]*name="([^"]+)"',
        text
    )

    if match:
        return match.group(1).strip()

    match = re.search(
        r'<identifier[^>]*name="([^"]+)"',
        text
    )

    if match:
        return match.group(1).strip()

    return None

# cirpy 
def extract_smiles_from_cirpy(compound_name):
    if compound_name is None:
        return None

    try: 
        smiles = cirpy.resolve(compound_name, "smiles")
        if smiles is not None and smiles.strip() != "":
            return smiles.strip()

    except:
        pass
    return None 

def clean_compound_name(compound_name):
    """
    Clean NP-MRD names before database lookup
    """

    if compound_name is None:
        return None

    name = compound_name.strip()

    # remove stereochemistry prefixes
    name = re.sub(r'^\(\+\)-', '', name)
    name = re.sub(r'^\(-\)-', '', name)
    name = re.sub(r'^\(R\)-', '', name)
    name = re.sub(r'^\(S\)-', '', name)

    # skip NP-MRD placeholder names
    if name.upper().startswith("NP") and "_" in name:
        return None

    # skip 13C names
    if "13C".upper() in name.upper():
        return None

    if "HSQC".upper() in name.upper():
        return None

    return name.strip()


# def extract_smiles_with_fallbacks(text, compound_name):
#     """
#     Try all SMILES methods without deleting old code
#     """

#     smiles = extract_smiles(text)

#     if smiles is None:
#         smiles = extract_smiles_from_structure(text)

#     clean_name = clean_compound_name(compound_name)

#     if smiles is None:
#         smiles = extract_smiles_from_pubchem(clean_name)

#     if smiles is None:
#         smiles = extract_smiles_from_cactus(clean_name)

#     if smiles is None:
#         smiles = extract_smiles_from_cirpy(clean_name)

#     return smiles

# lotus 

def extract_smiles_from_lotus(compound_name):

    if compound_name is None:
        return None

    compound_name = compound_name.strip().lower()

    return lotus_lookup.get(compound_name)



def extract_compound_id(filename, text):
    """
    get the compound ID from the file name of each file
    """

    match = re.search(r"NP\d+", filename, re.IGNORECASE)
    if match:
        return match.group(0).upper()

    match = re.search(r"NP\d+", text, re.IGNORECASE)
    if match:
        return match.group(0).upper()

    return os.path.splitext(filename)[0]


def read_text_file(path):
    encodings = ["utf-8", "latin-1", "cp1252"]

    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                return f.read()
        except:
            continue

    return ""

# lotus_lookup = load_lotus_lookup(lotus_file)
# print("LOTUS loaded:", len(lotus_lookup))

# with open(lotus_file, "r", encoding="utf-8", errors="ignore") as f:
#     for i in range(5):
#         print(f.readline())

# def load_structure_smiles_lookup(smiles_csv):
#     df = pd.read_csv(smiles_csv)

#     print("Structure columns:", df.columns.tolist())

#     id_col = "accession"
#     smiles_col = "smiles"

#     lookup = {}

#     for _, row in df.iterrows():
#         compound_id = str(row[id_col]).strip().upper()
#         smiles = row[smiles_col]

#         if pd.isna(smiles):
#             continue

#         lookup[compound_id] = str(smiles).strip()

#     return lookup

structure_smiles_lookup = load_structure_smiles_lookup(smiles_csv)
print("Structure SMILES loaded:", len(structure_smiles_lookup))

rows = []


for root, dirs, files in os.walk(download_folder):
    for filename in files:
        if len(rows) >= max_rows:
            break

        path = os.path.join(root, filename)
        ext = os.path.splitext(filename)[1].lower()

        peaks = []
        smiles = None
        compound_id = None
        source_type = ext

        if "13C".upper() in filename.upper():
            continue

        if "HSQC".upper() in filename.upper():
            continue

        # peak lists in csv, json, xml, and txt
        if ext in [".txt", ".csv", ".json", ".xml", ".nmrml"]:
            text = read_text_file(path)

            compound_id = extract_compound_id(filename, text)
            compound_name = extract_compound_name(text)
            smiles = structure_smiles_lookup.get(compound_id)


            smiles = extract_smiles(text)
            if smiles is None:
                smiles = extract_smiles_from_structure(text)
                print(compound_id, compound_name, smiles)

            # if smiles is None:
            #     smiles = extract_smiles_from_pubchem(compound_name)

            # # cactus
            # if smiles is None:
            #     smiles = extract_smiles_from_cactus(compound_name)

            # # if smiles is None:
            # #     smiles = extract_smiles_from_opsin(compound_name)

            # if smiles is None:
            #     smiles = extract_smiles_from_cirpy(compound_name)

            # # extra fallback using cleaned name
            # # if smiles is None:
            # #  smiles = extract_smiles_with_fallbacks(text, compound_name)

            # clean_name = clean_compound_name(compound_name)
            # if smiles is None:
            #     smiles = extract_smiles_from_lotus(clean_name)

            if compound_name is None:
                continue 

            if smiles is None:
                continue 

            peaks = parse_peak_text(text)
            

        elif ext in [".jdx", ".dx", ".jcamp"]:
            if jcamp_readfile is None:
                continue

            try:
                data = jcamp_readfile(path)
            except Exception as e:
                print("Could not read jcamp: ", filename, e)
                continue

            compound_id = os.path.splitext(filename)[0]
            

            if "x" in data and "y" in data:
                x_raw = np.array(data["x"], dtype=float)
                y_raw = np.array(data["y"], dtype=float)

                mask = np.isfinite(x_raw) & np.isfinite(y_raw)
                x_raw = x_raw[mask]
                y_raw = y_raw[mask]

                mask = (x_raw > xMin) & (x_raw <= xMax)
                x_raw = x_raw[mask]
                y_raw = y_raw[mask]

                if len(x_raw) < 2:
                    continue

                order = np.argsort(x_raw)
                x_raw = x_raw[order]
                y_raw = y_raw[order]

                x_new = np.linspace(xMin, xMax, points)
                y_new = np.interp(x_new, x_raw, y_raw, left=0, right=0)

                if np.max(np.abs(y_new)) > 0:
                    y_new = y_new / np.max(np.abs(y_new))

                rows.append({
                    "compound_id": compound_id,
                    "name": compound_name,
                    "smiles": smiles,
                    "peaks_ppm": None,
                    "x_ppm": json.dumps(x_new.tolist()),
                    "y_intensity": json.dumps(y_new.tolist()),
                    "source_file": filename,
                    "source_type": source_type
                })

                continue

        else:
            continue

        if len(peaks) < 1:
            continue

        x, y = make_spectrum(peaks, x_min=xMin, x_max=xMax, n=points)
        if smiles is None:
            print("SKIPPING NO SMILES: ", compound_id, compound_name)
            continue

        rows.append({
            "compound_id": compound_id,
            "name": compound_name,
            "smiles": smiles,
            "peaks_ppm": json.dumps([p[0] for p in peaks]),
            "x_ppm": json.dumps(x),
            "y_intensity": json.dumps(y),
            "source_file": filename,
            "source_type": source_type
        })

        print(
            "Collected:",
            len(rows),
            "|",
            compound_name,
            "|",
            smiles
        )

        if len(rows) % 1 == 0:
            print("Collected: ", len(rows))

    if len(rows) >= max_rows:
        break


df = pd.DataFrame(rows)

df = df[df["smiles"].notna()]
df = df[df["smiles"].astype(str).str.lower() != "none"]

df.to_csv(output_csv, index=False)

print("Saved CSV:", output_csv)
print("Rows:", len(df))
print(df.head())