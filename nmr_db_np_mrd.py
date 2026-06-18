import os
import re
import json
import numpy as np
import pandas as pd


download_folder = r"C:\Users\herna\Sode Labs\downloads"
output_csv = "np_mrd_1H_1000_expdata.csv"

max_rows = 1000
xMin = 0
xMax = 12
points = 2000

def make_spectrum(peaks, x_min=0, x_max=12, n=2000, width=0.025):
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
    lines = text.splitlines()
    for line in lines:
        line_lower = line.lower()

        # this skips obvious non peaks lines
        if any(word in line_lower for word in ["name", "smiles", "inchi", "formula", "copyright"]):
            continue

        nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
        if not nums:
            continue 

        try:
            shift = float(nums[0])
        except:
            continue 

        if not (0 <= shift <= 12):
            continue

        intensity = 1.0
        if len(nums) >= 2:
            try:
                possible_intensity = float(nums[1])
                if possible_intensity > 0:
                    intensity = possible_intensity 
            except:
                pass
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

        # peak lists in csv, json, xml, and txt
        if ext in [".txt", ".csv", ".json", ".xml", ".nmrml"]:
            text = read_text_file(path)

            compound_id = extract_compound_id(filename, text)
            smiles = extract_smiles(text)

            peak_area = text[text.find("peakList"):]

            peaks = parse_peak_text(peak_area)
            

           

            

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

        if len(peaks) < 2:
            continue

        x, y = make_spectrum(peaks, x_min=xMin, x_max=xMax, n=points)

        rows.append({
            "compound_id": compound_id,
            "smiles": smiles,
            "peaks_ppm": json.dumps([p[0] for p in peaks]),
            "x_ppm": json.dumps(x),
            "y_intensity": json.dumps(y),
            "source_file": filename,
            "source_type": source_type
        })

        if len(rows) % 100 == 0:
            print("Collected: ", len(rows))

    if len(rows) >= max_rows:
        break


df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)

print("Saved CSV:", output_csv)
print("Rows:", len(df))
print(df.head())
