# import os
# import re
# import json
# import numpy as np
# import pubchempy as pcp
# import pandas as pd
# from rdkit import Chem
# from xml.etree import ElementTree as ET


# download_folder = r"C:\Users\herna\Sode Labs\downloads"
# output_csv = "np_mrd_1H_250_expdata.csv"

# max_rows = 250
# xMin = 0
# xMax = 12
# points = 1000

# def make_spectrum(peaks, x_min=0, x_max=12, n=1000, width=0.025):
#     x = np.linspace(x_min, x_max, n)
#     y = np.zeros_like(x)

#     for shift, intensity in peaks:
#         y += intensity * np.exp(-0.5 * ((x - shift) / width) ** 2)

#     if y.max() > 0:
#         y = y / y.max()
#     return x.tolist(), y.tolist()

# def parse_peak_text(text):
#     """
#     Extract 1H NMR peaks from the NP MRD database ----- csv files
#     Takes from 0 to 12 ppm peaks and overall data 
#     """

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

#     # this removes duplicate shifts 
#     unique = {}
#     for shift, intensity in peaks:
#         unique[round(shift, 4)] = max(unique.get(round(shift, 4), 0), intensity)

#     return [(shift, intensity) for shift, intensity in unique.items()]
# def extract_smiles(text):
#     """
#     Find smiles inside the downloaded files
#     """

#     patterns = [
#         r'"smiles"\s*:\s*([^"]+)"',
#         r"<smiles>(.*?)</smiles>",
#         r"SMILES\s*[:=]\s*([^\s,;]+)",
#         r"smiles\s*[:=]\s*([^\s,;]+)"
#     ]

#     for pattern in patterns:
#         match = re.search(pattern, text, re.IGNORECASE)
#         if match:
#             return match.group(1).strip()

#     return None

# def extract_smiles_from_structure(text):
#     """
#     build a smiles string from the atomlist and bondlist 
#     """
#     try:
#         root = ET.fromstring(text)
#     except:
#         return None

#     ns = {"nmr": "http://nmrml.org/schema"}

#     atom_dict = {}
#     mol = Chem.RWMol()

#     # read the atoms 
#     for atom in root.findall(".//nmr:atomList/nmr:atom", ns):
#         atom_id = atom.attrib.get("id")
#         element = atom.attrib.get("elementType")

#         if atom_id is None or element is None:
#             continue

#         if element == "H":
#             continue 
            
#         try:
#             rd_atom = Chem.Atom(element)
#             idx = mol.AddAtom(rd_atom)
#             atom_dict[atom_id] = idx
#         except:
#             continue

#     # read the bonds
#     for bond in root.findall(".//nmr:bondList/nmr:bond", ns):
#         refs = bond.attrib.get("atomRefs")
#         order = bond.attrib.get("order", "1")

#         if refs is None:
#             continue

#         parts = refs.split()
#         if len(parts) != 2:
#             continue 

#         a1, a2 = parts
#         if a1 not in atom_dict or a2 not in atom_dict:
#             continue 

#         if order == "1":
#             bond_type = Chem.BondType.SINGLE

#         elif order == "2":
#             bond_type = Chem.BondType.DOUBLE
#         elif order == "3":
#             bond_type = Chem.BondType.TRIPLE
#         else:
#             bond_type = Chem.BondType.SINGLE
#         try:
#             mol.AddBond(atom_dict[a1], atom_dict[a2], bond_type)
#         except:
#             continue

#     try:
#         final_mol = mol.GetMol()
#         Chem.SanitizeMol(final_mol)
#         smiles = Chem.MolToSmiles(final_mol)
#         return smiles
#     except:
#         return None 

# def extract_smiles_from_pubchem(compound_name):
#     """
#     uses pubchem as an alt
#     """
#     if compound_name is None:
#         return None

#     try:
#         results = pcp.get_compounds(compound_name, "name")
#         if len(results) == 0:
#             return None 

#         return results[0].canonical.smiles
#     except:
#         return None

    

# def extract_compound_name(text):
#     """ 
#     This extracts the compound name from the nmrML file 
#     """
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

# def extract_compound_id(filename, text):
#     """
#     get the compound ID from the file name of each file
#     """

#     match = re.search(r"NP\d+", filename, re.IGNORECASE)
#     if match:
#         return match.group(0).upper()

#     match = re.search(r"NP\d+", text, re.IGNORECASE)
#     if match:
#         return match.group(0).upper()

#     return os.path.splitext(filename)[0]


# def read_text_file(path):
#     encodings = ["utf-8", "latin-1", "cp1252"]

#     for enc in encodings:
#         try:
#             with open(path, "r", encoding=enc, errors="ignore") as f:
#                 return f.read()
#         except:
#             continue

#     return ""


# rows = []


# for root, dirs, files in os.walk(download_folder):
#     for filename in files:
#         if len(rows) >= max_rows:
#             break

#         path = os.path.join(root, filename)
#         ext = os.path.splitext(filename)[1].lower()

#         peaks = []
#         smiles = None
#         compound_id = None
#         source_type = ext

#         # peak lists in csv, json, xml, and txt
#         if ext in [".txt", ".csv", ".json", ".xml", ".nmrml"]:
#             text = read_text_file(path)

#             compound_id = extract_compound_id(filename, text)
#             compound_name = extract_compound_name(text)

#             smiles = extract_smiles(text)
#             if smiles is None:
#                 smiles = extract_smiles_from_structure(text)
#                 print(compound_id, compound_name, smiles)

#             if smiles is None:
#                 smiles = extract_smiles_from_pubchem(compound_name)

#             if compound_name is None:
#                 continue 

#             if smiles is None:
#                 continue 

#             peaks = parse_peak_text(text)
            

#         elif ext in [".jdx", ".dx", ".jcamp"]:
#             if jcamp_readfile is None:
#                 continue

#             try:
#                 data = jcamp_readfile(path)
#             except Exception as e:
#                 print("Could not read jcamp: ", filename, e)
#                 continue

#             compound_id = os.path.splitext(filename)[0]
            

#             if "x" in data and "y" in data:
#                 x_raw = np.array(data["x"], dtype=float)
#                 y_raw = np.array(data["y"], dtype=float)

#                 mask = np.isfinite(x_raw) & np.isfinite(y_raw)
#                 x_raw = x_raw[mask]
#                 y_raw = y_raw[mask]

#                 mask = (x_raw > xMin) & (x_raw <= xMax)
#                 x_raw = x_raw[mask]
#                 y_raw = y_raw[mask]

#                 if len(x_raw) < 2:
#                     continue

#                 order = np.argsort(x_raw)
#                 x_raw = x_raw[order]
#                 y_raw = y_raw[order]

#                 x_new = np.linspace(xMin, xMax, points)
#                 y_new = np.interp(x_new, x_raw, y_raw, left=0, right=0)

#                 if np.max(np.abs(y_new)) > 0:
#                     y_new = y_new / np.max(np.abs(y_new))

#                 rows.append({
#                     "compound_id": compound_id,
#                     "name": compound_name,
#                     "smiles": smiles,
#                     "peaks_ppm": None,
#                     "x_ppm": json.dumps(x_new.tolist()),
#                     "y_intensity": json.dumps(y_new.tolist()),
#                     "source_file": filename,
#                     "source_type": source_type
#                 })

#                 continue

#         else:
#             continue

#         if len(peaks) < 2:
#             continue

#         x, y = make_spectrum(peaks, x_min=xMin, x_max=xMax, n=points)

#         rows.append({
#             "compound_id": compound_id,
#             "name": compound_name,
#             "smiles": smiles,
#             "peaks_ppm": json.dumps([p[0] for p in peaks]),
#             "x_ppm": json.dumps(x),
#             "y_intensity": json.dumps(y),
#             "source_file": filename,
#             "source_type": source_type
#         })

#         if len(rows) % 1 == 0:
#             print("Collected: ", len(rows))

#     if len(rows) >= max_rows:
#         break


# df = pd.DataFrame(rows)
# df.to_csv(output_csv, index=False)

# print("Saved CSV:", output_csv)
# print("Rows:", len(df))
# print(df.head())


