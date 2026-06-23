import os 
import re
import pandas as pd 

csv = "np_mrd_1H_1500_expdata.csv"
download_folder = r"C:\Users\herna\Sode Labs\downloads"
folder = "NMR SPECTRAL DETAILS"
os.makedirs(folder, exist_ok=True)

def safe_filename(text):
    text = str(text)
    for bad in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
        text = text.replace(bad, "_")
    return text

def read_text_file(path):
    encodings = ["utf-8", "latin-1", "cp1252"]

    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                return f.read()
        except:
            continue 
    return ""

def build_file_lookup(download_folder):
    file_lookup = {}
    for root, dirs, files in os.walk(download_folder):
        for filename in files:
            file_lookup[filename] = os.path.join(root, filename)

    return file_lookup

def extract_nmr_details_from_nmrml(text):
    details = []
    multiplet_blocks = re.findall(
        r'<multiplet[^>]*center="([^"]+)"[^>]*>(.*?)</multiplet>',
        text,
        re.DOTALL
    )
    for multiplet_center, block in multiplet_blocks:
        try:
            center_ppm = float(multiplet_center)
        except:
            center_ppm = None

        atom_match = re.search(
            r'<atoms[^>]*atomRefs="([^"]+)"',
            block
        )
        if atom_match:
            atom_refs = atom_match.group(1).split()
            integration = f"{len(atom_refs)}H"
        else:
            atom_refs = []
            integration = "unknown"

        multiplicity_match = re.search(
            r'<multiplicity[^>]*name="([^"]+)"',
            block
        )
        if multiplicity_match:
            peak_type = multiplicity_match.group(1)
            peak_type = peak_type.replace(" feature", "").strip()
        else:
            peak_type = "multiplet"

        peak_matches = re.findall(
            r'<peak[^>]*amplitude="([^"]+)"[^>]*center="([^"]+)"[^>]*width="([^"]+)"',
            block
        )

        if len(peak_matches) == 0:
            details.append({
                "ppm": center_ppm,
                "intensity": integration, 
                "peak_type": peak_type,
                "hz": "not available",
                "atom_refs": " ".join(atom_refs)
            })

        for amplitude, peak_center, width in peak_matches:
            try:
                ppm = float(peak_center)
            except:
                ppm = center_ppm

            try:
                amp = float(amplitude)
            except:
                amp = "unknown"

            try:
                hz = float(width)
            except:
                hz = "not available"

            details.append({
                "ppm": ppm, 
                "intensity": integration,
                "relative_amplitude": amp,
                "peak_type": peak_type,
                "hz": hz,
                "atom_refs": " ".join(atom_refs)
            })
    return details

df = pd.read_csv(csv)
file_lookup = build_file_lookup(download_folder)
print("Files indexed: ", len(file_lookup))
for i, row in df.iterrows():
    compound_id = row["compound_id"]
    compound_name = row["name"]
    smiles = row["smiles"]
    source_file = row["source_file"]
    output_name = safe_filename(f"{compound_id}_{compound_name}.txt")
    output_path = os.path.join(folder, output_name)

    nmrml_path = file_lookup.get(source_file)
    details = []

    if nmrml_path is not None:
        text = read_text_file(nmrml_path)
        details = extract_nmr_details_from_nmrml(text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Compound ID: {compound_id}\n")
        f.write(f"Name: {compound_name}\n")
        f.write(f"SMILES: {smiles}\n")
        f.write(f"Source file: {source_file}\n")
        f.write("\n")
        if nmrml_path is None:
            f.write("Original nmrML file not found.\n")
            continue
        if len(details) == 0:
            f.write("No multiplet/peak annotation found in the file\n")
            continue

        f.write("ppm\tintensity\tpeak_type\thz\trelative_amplitude\tatom_refs\n")

        for d in details:
            f.write(
                f"{d.get('ppm')}\t"
                f"{d.get('intensity')}\t"
                f"{d.get('peak_type')}\t"
                f"{d.get('hz')}\t"
                f"{d.get('relative_amplitude', 'not available')}\t"
                f"{d.get('atom_refs')}\n" 
            )

    if (i + 1) % 1 == 0:
        print("Saved: ", i + 1)            

print("Finished")
print("Details saved to: ", folder)
