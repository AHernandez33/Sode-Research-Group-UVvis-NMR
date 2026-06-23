import os
import re
import json
import pandas as pd
from rdkit import Chem

csv = "nmrshift_db2_1H_1500_expdata.csv"
sdf_file = r"C:\Users\herna\Sode Labs\NMR\nmrshiftdb2withsignals.sd"

folder = "NMR SPECTRAL DETAILS EXP 2"
os.makedirs(folder, exist_ok=True)


def safe_filename(text):
    text = str(text)
    for bad in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
        text = text.replace(bad, "_")
    return text


def clean_peak_type(text):
    text = str(text).lower().strip()

    replacements = {
        "s": "singlet",
        "d": "doublet",
        "t": "triplet",
        "q": "quartet",
        "m": "multiplet",
        "dd": "doublet of doublets",
        "td": "triplet of doublets",
        "dt": "doublet of triplets",
    }

    return replacements.get(text, text)


def safe_float(value, default="not available"):
    value = str(value).strip()
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    if match:
        return float(match.group(0))
    return default


def parse_spectrum_text(raw_text):
    peaks = []

    raw_text = str(raw_text)

    # Split into lines or semicolon blocks
    lines = re.split(r"[\n\r|]+", raw_text)

    for line in lines:
        line = line.strip()

        if not line:
            continue

        parts = line.split(";")

        if len(parts) >= 3:
            ppm = safe_float(parts[0])
            # relative_amplitude = safe_float(parts[1])
            atom_refs = str(parts[2].strip())

            if ppm == "not available":
                continue

            peaks.append({
                "ppm": ppm,
                # "relative_amplitude": relative_amplitude,
                "atom_refs": atom_refs
            })

        # # Find ppm values between 0 and 12
        # ppm_matches = re.findall(r"\b([0-9]+(?:\.[0-9]+)?)\b", line)

        # if len(ppm_matches) == 0:
        #     continue

        # ppm = None
        # for value in ppm_matches:
        #     value_float = float(value)
        #     if 0 <= value_float <= 12:
        #         ppm = value_float
        #         break

        # if ppm is None:
        #     continue

        # # Try to find H count like 1H, 2H, 3H
        # intensity_match = re.search(r"\b([0-9]+H)\b", line)
        # intensity = intensity_match.group(1) if intensity_match else "1H"

        # # Try to find multiplicity
        # peak_type = "not available"

        # mult_patterns = [
        #     r"\b(doublet of doublets)\b",
        #     r"\b(triplet of doublets)\b",
        #     r"\b(doublet of triplets)\b",
        #     r"\b(singlet)\b",
        #     r"\b(doublet)\b",
        #     r"\b(triplet)\b",
        #     r"\b(quartet)\b",
        #     r"\b(multiplet)\b",
        #     r"\b(dd)\b",
        #     r"\b(td)\b",
        #     r"\b(dt)\b",
        #     r"\b(s)\b",
        #     r"\b(d)\b",
        #     r"\b(t)\b",
        #     r"\b(q)\b",
        #     r"\b(m)\b",
        # ]

        # for pattern in mult_patterns:
        #     match = re.search(pattern, line, flags=re.IGNORECASE)
        #     if match:
        #         peak_type = clean_peak_type(match.group(1))
        #         break

        # # Hz value: use ppm * 500 if unavailable
        # # Your example looks like 6.43 ppm -> 3215 Hz, so likely 500 MHz instrument
        # hz = round(ppm * 500, 1)

        # # Relative amplitude: unavailable unless stored in raw text
        # relative_amplitude = "not available"

        # # Atom refs
        # atom_refs = "not available"

        # atom_match = re.search(r"(?:atom|atoms|atom_refs|assignment)[^\d]*(\d+(?:[, ]+\d+)*)", line, flags=re.IGNORECASE)
        # if atom_match:
        #     atom_refs = atom_match.group(1).strip()

        # peaks.append({
        #     "ppm": ppm,
        #     "intensity": intensity,
        #     "peak_type": peak_type,
        #     "hz": hz,
        #     "relative_amplitude": relative_amplitude,
        #     "atom_refs": atom_refs

    return peaks


def build_sdf_lookup(sdf_file):
    lookup = {}

    supplier = Chem.SDMolSupplier(sdf_file, removeHs=False)

    for mol_index, mol in enumerate(supplier):
        if mol is None:
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

        compound_id = str(compound_id)

        spectrum_texts = []

        for key in props.keys():
            if "Spectrum 1H" in key:
                spectrum_texts.append({
                    "key": key,
                    "text": str(props[key])
                })

        lookup[compound_id] = spectrum_texts

    return lookup


df = pd.read_csv(csv)

sdf_lookup = build_sdf_lookup(sdf_file)

print("Rows in CSV:", len(df))
print("SDF compounds indexed:", len(sdf_lookup))

for i, row in df.iterrows():

    compound_id = str(row["compound_id"])
    compound_name = str(row["name"])
    smiles = str(row["smiles"])

    output_name = safe_filename(f"{compound_id}_{compound_name}.txt")
    output_path = os.path.join(folder, output_name)

    spectrum_texts = sdf_lookup.get(compound_id, [])

    all_peak_details = []

    for spec in spectrum_texts:
        parsed_peaks = parse_spectrum_text(spec["text"])
        all_peak_details.extend(parsed_peaks)

    with open(output_path, "w", encoding="utf-8") as f:

        f.write(f"Compound ID: {compound_id}\n")
        f.write(f"Name: {compound_name}\n")
        f.write(f"SMILES: {smiles}\n")
        f.write(f"Source: nmrshiftdb2withsignals.sd\n")
        f.write("\n")

        f.write("ppm\tatom_ref\n")
        # f.write("ppm\trelative_amplitude\tatom_refs\n")

        if len(all_peak_details) == 0:
            f.write("No detailed peaks found.\n")
        else:
            for peak in all_peak_details:
                f.write(
                    f"{peak['ppm']}\t"
                    # f"{peak['relative_amplitude']}\t"
                    f"{peak['atom_refs']}\n"
                )

        f.write("\n")
        f.write("Raw nmrshiftdb2 1H spectrum text:\n")
        f.write("=" * 60 + "\n")

        if len(spectrum_texts) == 0:
            f.write("No raw 1H spectrum text found in SDF.\n")
        else:
            for spec in spectrum_texts:
                f.write(f"\nProperty key: {spec['key']}\n")
                f.write("-" * 60 + "\n")
                f.write(spec["text"])
                f.write("\n")

    print(f"Saved {i + 1}/{len(df)}: {compound_id} | {compound_name}")

print("Finished")
print("Details saved to:", folder)